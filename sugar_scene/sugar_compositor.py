import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from sugar_scene.cameras import CamerasWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.graphics_utils import *

# This script is still in development. 
# We used a less clean version of this script for the paper.

class SuGaRCompositor(nn.Module):
    """Class to perform composition and animation of multiple SuGaR models.
    """
    def __init__(
        self,
        sugar_models:list[SuGaR],
        idx_to_render:list[torch.Tensor],
        *args, **kwargs) -> None:
        super(SuGaR, self).__init__()
        
        self.sugar_models = nn.ModuleList(sugar_models)
        self.idx_to_render = idx_to_render
        
    @property
    def device(self):
        return self.sugar_models[0].device()
    
    def render_image_gaussian_rasterizer(
        self, 
        nerf_cameras:CamerasWrapper=None, 
        camera_indices:int=0,
        verbose=False,
        bg_color = None,
        sh_deg:int=None,
        sh_rotations:list[torch.Tensor]=None,
        compute_color_in_rasterizer=False,
        compute_covariance_in_rasterizer=True,
        return_2d_radii = False,
        quaternions=None,
        use_same_scale_in_all_directions=False,
        return_opacities:bool=False,
        return_colors:bool=False,
        positions:torch.Tensor=None,
        ):
        """Render an image using the Gaussian Splatting Rasterizer.

        Args:
            nerf_cameras (CamerasWrapper, optional): _description_. Defaults to None.
            camera_indices (int, optional): _description_. Defaults to 0.
            verbose (bool, optional): _description_. Defaults to False.
            bg_color (_type_, optional): _description_. Defaults to None.
            sh_deg (int, optional): _description_. Defaults to None.
            sh_rotations (torch.Tensor, optional): _description_. Defaults to None.
            compute_color_in_rasterizer (bool, optional): _description_. Defaults to False.
            compute_covariance_in_rasterizer (bool, optional): _description_. Defaults to True.
            return_2d_radii (bool, optional): _description_. Defaults to False.
            quaternions (_type_, optional): _description_. Defaults to None.
            use_same_scale_in_all_directions (bool, optional): _description_. Defaults to False.
            return_opacities (bool, optional): _description_. Defaults to False.
            return_colors (bool, optional): _description_. Defaults to False.
            positions (torch.Tensor, optional): _description_. Defaults to None.
            point_colors (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if nerf_cameras is None:
            nerf_cameras = self.sugar_models[0].nerfmodel.training_cameras

        p3d_camera = nerf_cameras.p3d_cameras[camera_indices]

        if bg_color is None:
            bg_color = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)

        use_torch = False
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = nerf_cameras.camera_to_worlds[camera_indices]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(self.device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        world_view_transform = torch.Tensor(getWorld2View(
            R=R, t=T, tensor=use_torch)).transpose(0, 1).cuda()
        
        proj_transform = getProjectionMatrix(
            p3d_camera.znear.item(), 
            p3d_camera.zfar.item(), 
            nerf_cameras.gs_cameras[0].FoVx, 
            nerf_cameras.gs_cameras[0].FoVy).transpose(0, 1).cuda()
        # TODO: THE TWO FOLLOWING LINES ARE IMPORTANT! IT'S NOT HERE IN 3DGS CODE! Should make a PR when I have time
        proj_transform[..., 2, 0] = - p3d_camera.K[0, 0, 2]
        proj_transform[..., 2, 1] = - p3d_camera.K[0, 1, 2]
        
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_transform.unsqueeze(0))).squeeze(0)

        camera_center = p3d_camera.get_camera_center()
        if verbose:
            print("p3d camera_center", camera_center)
            print("ns camera_center", nerf_cameras.camera_to_worlds[camera_indices][..., 3])

        raster_settings = GaussianRasterizationSettings(
            image_height=int(nerf_cameras.height),
            image_width=int(nerf_cameras.width),
            tanfovx=math.tan(nerf_cameras.gs_cameras[0].FoVx * 0.5),
            tanfovy=math.tan(nerf_cameras.gs_cameras[0].FoVy * 0.5),
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_deg,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
    
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # TODO: Change color computation to match 3DGS paper (remove sigmoid)
        if not compute_color_in_rasterizer:
            shs = None
            splat_colors = torch.zeros(0, 3, dtype=torch.float, device=self.device)
        else:
            shs = torch.zeros(0, (sh_deg+1)**2, 3, dtype=torch.float, device=self.device)
            splat_colors = None
        if quaternions is None:
            quaternions = torch.zeros(0, 4, dtype=torch.float, device=self.device)
        scales = torch.zeros(0, 3, dtype=torch.float, device=self.device)
        splat_opacities = torch.zeros(0, 1, dtype=torch.float, device=self.device)
        
        for i_model, sugar_model in enumerate(self.sugar_models):
            idx_to_render = self.idx_to_render[i_model]
            
            if not compute_color_in_rasterizer:
                if sh_rotations is None or sh_rotations[i_model] is None:
                    splat_colors_i = sugar_model.get_points_rgb( 
                        camera_centers=camera_center,
                        sh_levels=sh_deg+1,)
                else:
                    splat_colors_i = sugar_model.get_points_rgb(
                        camera_centers=None,
                        directions=(torch.nn.functional.normalize(positions - camera_center, dim=-1).unsqueeze(1) @ sh_rotations[i_model])[..., 0, :],
                        sh_levels=sh_deg+1,)
                splat_colors_i = splat_colors_i[idx_to_render]
                splat_colors = torch.cat([splat_colors, splat_colors_i], dim=0)
            else:
                shs_i = sugar_model.sh_coordinates[idx_to_render]
                shs = torch.cat([shs, shs_i], dim=0)
            
            splat_opacities_i = sugar_model.strengths.view(-1, 1)[idx_to_render]
            splat_opacities = torch.cat([splat_opacities, splat_opacities_i], dim=0)
            
            if quaternions is None:
                quaternions_i = sugar_model.quaternions[idx_to_render]
                quaternions = torch.cat([quaternions, quaternions_i], dim=0)
            
            if not use_same_scale_in_all_directions:
                scales_i = sugar_model.scaling[idx_to_render]
            else:
                scales_i = sugar_model.scaling.mean(dim=-1, keepdim=True).expand(-1, 3)[idx_to_render]
            scales = torch.cat([scales, scales_i], dim=0)
        
        if verbose:
            print("Scales:", scales.shape, scales.min(), scales.max())

        if not compute_covariance_in_rasterizer:            
            cov3Dmatrix = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device=self.device)
            rotation = quaternion_to_matrix(quaternions)

            cov3Dmatrix[:,0,0] = scales[:,0]**2
            cov3Dmatrix[:,1,1] = scales[:,1]**2
            cov3Dmatrix[:,2,2] = scales[:,2]**2
            cov3Dmatrix = rotation @ cov3Dmatrix @ rotation.transpose(-1, -2)
            # cov3Dmatrix = rotation @ cov3Dmatrix
            
            cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)

            cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
            cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
            cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
            cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
            cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
            cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
            
            quaternions = None
            scales = None
        else:
            cov3D = None
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(self._points, dtype=self._points.dtype, requires_grad=True, device=self.device) + 0
        screenspace_points = torch.zeros(len(scales), 3, dtype=scales.dtype, requires_grad=True, device=self.device)
        if return_2d_radii:
            try:
                screenspace_points.retain_grad()
            except:
                print("WARNING: return_2d_radii is True, but failed to retain grad of screenspace_points!")
                pass
        means2D = screenspace_points
        
        if verbose:
            print("points", positions.shape)
            if not compute_color_in_rasterizer:
                print("splat_colors", splat_colors.shape)
            print("splat_opacities", splat_opacities.shape)
            if not compute_covariance_in_rasterizer:
                print("cov3D", cov3D.shape)
                print(cov3D[0])
            else:
                print("quaternions", quaternions.shape)
                print("scales", scales.shape)
            print("screenspace_points", screenspace_points.shape)
        
        rendered_image, radii = rasterizer(
            means3D = positions,
            means2D = means2D,
            shs = shs,
            colors_precomp = splat_colors,
            opacities = splat_opacities,
            scales = scales,
            rotations = quaternions,
            cov3D_precomp = cov3D)
        
        if not(return_2d_radii or return_opacities or return_colors):
            return rendered_image.transpose(0, 1).transpose(1, 2)
        
        else:
            outputs = {
                "image": rendered_image.transpose(0, 1).transpose(1, 2),
                "radii": radii,
                "viewspace_points": screenspace_points,
            }
            if return_opacities:
                outputs["opacities"] = splat_opacities
            if return_colors:
                outputs["colors"] = splat_colors
        
            return outputs
        