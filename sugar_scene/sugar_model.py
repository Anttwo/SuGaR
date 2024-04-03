import torch.nn as nn
from torch.nn.functional import normalize as torch_normalize
import open3d as o3d
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_apply, quaternion_invert, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from sugar_utils.spherical_harmonics import (
    eval_sh, RGB2SH, SH2RGB,
)
from sugar_utils.graphics_utils import *
from sugar_utils.general_utils import inverse_sigmoid
from sugar_scene.gs_model import GaussianSplattingWrapper, GaussianModel
from sugar_scene.cameras import CamerasWrapper


scale_activation = torch.exp
scale_inverse_activation = torch.log
use_old_method = False


def _initialize_radiuses_gauss_rasterizer(sugar):
    """Function to initialize the  of a SuGaR model.

    Args:
        sugar (SuGaR): SuGaR model.

    Returns:
        Tensor: Tensor with shape (n_points, 4+3) containing 
            the initial quaternions and scaling factors.
    """
    # Initialize learnable radiuses
    sugar.image_height = int(sugar.nerfmodel.training_cameras.height[0].item())
    sugar.image_width = int(sugar.nerfmodel.training_cameras.width[0].item())
    
    all_camera_centers = sugar.nerfmodel.training_cameras.camera_to_worlds[..., 3]
    all_camera_dists = torch.cdist(sugar.points, all_camera_centers)[None]
    d_charac = all_camera_dists.mean(-1, keepdim=True)
    
    ndc_factor = 1.
    sugar.min_ndc_radius = ndc_factor * 2. / min(sugar.image_height, sugar.image_width)
    sugar.max_ndc_radius = ndc_factor * 2. * 0.05  # 2. * 0.01
    sugar.min_radius = sugar.min_ndc_radius / sugar.focal_factor * d_charac
    sugar.max_radius = sugar.max_ndc_radius / sugar.focal_factor * d_charac
    
    knn = knn_points(sugar.points[None], sugar.points[None], K=4)
    use_sqrt = True
    use_mean = False
    initial_radius_normalization = 1.  # 1., 0.1
    if use_sqrt:
        knn_dists = torch.sqrt(knn.dists[..., 1:])
    else:
        knn_dists = knn.dists[..., 1:]
    if use_mean:
        print("Use mean to initialize scales.")
        radiuses = knn_dists.mean(-1, keepdim=True).clamp_min(0.0000001) * initial_radius_normalization
    else:
        print("Use min to initialize scales.")
        radiuses = knn_dists.min(-1, keepdim=True)[0].clamp_min(0.0000001) * initial_radius_normalization
    
    res = inverse_radius_fn(radiuses=radiuses)
    sugar.radius_dim = res.shape[-1]
    
    return res


def radius_fn(radiuses:torch.Tensor, max_value=0.2):
    scales = scale_activation(radiuses[..., 4:])
    return (scales.abs().clamp(max=max_value).max(dim=-1, keepdim=True)[0])
    
    
def inverse_radius_fn(radiuses:torch.Tensor):
    scales = scale_inverse_activation(radiuses.expand(-1, -1, 3).clone())
    quaternions = matrix_to_quaternion(
        torch.eye(3)[None, None].repeat(1, radiuses.shape[1], 1, 1).to(radiuses.device)
        )
    return torch.cat([quaternions, scales], dim=-1)


class SuGaR(nn.Module):
    """Main class for SuGaR models.
    Because SuGaR optimization starts with first optimizing a vanilla Gaussian Splatting model for 7k iterations,
    we built this class as a wrapper of a vanilla Gaussian Splatting model.
    Consequently, a corresponding Gaussian Splatting model trained for 7k iterations must be provided.
    However, this wrapper implementation may not be the most optimal one for memory usage, so we might change it in the future.
    """
    def __init__(
        self, 
        nerfmodel: GaussianSplattingWrapper,
        points: torch.Tensor,
        colors: torch.Tensor,
        initialize:bool=True,
        sh_levels:int=4,
        learnable_positions:bool=True,
        triangle_scale:float=2.,
        keep_track_of_knn:bool=False,
        knn_to_track:int=16,
        learn_color_only=False,
        beta_mode='average',  # 'learnable', 'average', 'weighted_average'
        freeze_gaussians=False,
        primitive_types='diamond',  # 'diamond', 'square'
        surface_mesh_to_bind=None,  # Open3D mesh
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=True,
        learn_surface_mesh_opacity=True,
        learn_surface_mesh_scales=True,
        n_gaussians_per_surface_triangle=6,  # 1, 3, 4 or 6
        editable=False,  # If True, allows for automatically rescaling Gaussians in real time if triangles are deformed from their original shape. 
        # We wrote about this functionality in the paper, and it was previously part of sugar_compositor.py, which we haven't finished cleaning yet for this repo.
        # We now moved it to this script as it is more related to the SuGaR model than to the compositor.
        *args, **kwargs) -> None:
        """
        Args:
            nerfmodel (GaussianSplattingWrapper): A vanilla Gaussian Splatting model trained for 7k iterations.
            points (torch.Tensor): Initial positions of the Gaussians (not used when wrapping).
            colors (torch.Tensor): Initial colors of the Gaussians (not used when wrapping).
            initialize (bool, optional): Whether to initialize the radiuses. Defaults to True.
            sh_levels (int, optional): Number of spherical harmonics levels to use for the color features. Defaults to 4.
            learnable_positions (bool, optional): Whether to learn the positions of the Gaussians. Defaults to True.
            triangle_scale (float, optional): Scale of the triangles used to replace the Gaussians. Defaults to 2.
            keep_track_of_knn (bool, optional): Whether to keep track of the KNN information for training regularization. Defaults to False.
            knn_to_track (int, optional): Number of KNN to track. Defaults to 16.
            learn_color_only (bool, optional): Whether to learn only the color features. Defaults to False.
            beta_mode (str, optional): Whether to use a learnable beta, or to average the beta values. Defaults to 'average'.
            freeze_gaussians (bool, optional): Whether to freeze the Gaussians. Defaults to False.
            primitive_types (str, optional): Type of primitive to use to replace the Gaussians. Defaults to 'diamond'.
            surface_mesh_to_bind (None, optional): Surface mesh to bind the Gaussians to. Defaults to None.
            surface_mesh_thickness (None, optional): Thickness of the bound Gaussians. Defaults to None.
            learn_surface_mesh_positions (bool, optional): Whether to learn the positions of the bound Gaussians. Defaults to True.
            learn_surface_mesh_opacity (bool, optional): Whether to learn the opacity of the bound Gaussians. Defaults to True.
            learn_surface_mesh_scales (bool, optional): Whether to learn the scales of the bound Gaussians. Defaults to True.
            n_gaussians_per_surface_triangle (int, optional): Number of bound Gaussians per surface triangle. Defaults to 6.
        """
        
        super(SuGaR, self).__init__()
        
        self.nerfmodel = nerfmodel
        self.freeze_gaussians = freeze_gaussians
        
        self.learn_positions = ((not learn_color_only) and learnable_positions) and (not freeze_gaussians)
        self.learn_opacities = (not learn_color_only) and (not freeze_gaussians)
        self.learn_scales = (not learn_color_only) and (not freeze_gaussians)
        self.learn_quaternions = (not learn_color_only) and (not freeze_gaussians)
        self.learnable_positions = learnable_positions
        
        if surface_mesh_to_bind is not None:
            self.learn_surface_mesh_positions = learn_surface_mesh_positions
            self.binded_to_surface_mesh = True
            self.learn_surface_mesh_opacity = learn_surface_mesh_opacity
            self.learn_surface_mesh_scales = learn_surface_mesh_scales
            self.n_gaussians_per_surface_triangle = n_gaussians_per_surface_triangle
            self.editable = editable
            
            self.learn_positions = self.learn_surface_mesh_positions
            self.learn_scales = self.learn_surface_mesh_scales
            self.learn_quaternions = self.learn_surface_mesh_scales
            self.learn_opacities = self.learn_surface_mesh_opacity
            
            self._surface_mesh_faces = torch.nn.Parameter(
                torch.tensor(np.array(surface_mesh_to_bind.triangles)).to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            if surface_mesh_thickness is None:
                surface_mesh_thickness = nerfmodel.training_cameras.get_spatial_extent() / 1_000_000
            self.surface_mesh_thickness = torch.nn.Parameter(
                torch.tensor(surface_mesh_thickness).to(nerfmodel.device), 
                requires_grad=False).to(nerfmodel.device)
            
            print("Binding radiance cloud to surface mesh...")
            if n_gaussians_per_surface_triangle == 1:
                self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/3, 1/3, 1/3]],
                    dtype=torch.float32,
                    device=nerfmodel.device,
                )[..., None]
            
            if n_gaussians_per_surface_triangle == 3:
                self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/2, 1/4, 1/4],
                    [1/4, 1/2, 1/4],
                    [1/4, 1/4, 1/2]],
                    dtype=torch.float32,
                    device=nerfmodel.device,
                )[..., None]
            
            if n_gaussians_per_surface_triangle == 4:
                self.surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/3, 1/3, 1/3],
                    [2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3]],
                    dtype=torch.float32,
                    device=nerfmodel.device,
                )[..., None]  # n_gaussians_per_face, 3, 1
                
            if n_gaussians_per_surface_triangle == 6:
                self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
                self.surface_triangle_bary_coords = torch.tensor(
                    [[2/3, 1/6, 1/6],
                    [1/6, 2/3, 1/6],
                    [1/6, 1/6, 2/3],
                    [1/6, 5/12, 5/12],
                    [5/12, 1/6, 5/12],
                    [5/12, 5/12, 1/6]],
                    dtype=torch.float32,
                    device=nerfmodel.device,
                )[..., None]
                
            points = torch.tensor(np.array(surface_mesh_to_bind.vertices)).float().to(nerfmodel.device)
            # verts_normals = torch.tensor(np.array(surface_mesh_to_bind.vertex_normals)).float().to(nerfmodel.device)
            self._vertex_colors = torch.tensor(np.array(surface_mesh_to_bind.vertex_colors)).float().to(nerfmodel.device)
            faces_colors = self._vertex_colors[self._surface_mesh_faces]  # n_faces, 3, n_coords
            colors = faces_colors[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_colors
            colors = colors.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_colors
            colors = colors.reshape(-1, 3)  # n_faces * n_gaussians_per_face, n_colors
                
            self._points = nn.Parameter(points, requires_grad=self.learn_positions).to(nerfmodel.device)
            n_points = len(np.array(surface_mesh_to_bind.triangles)) * n_gaussians_per_surface_triangle
            self._n_points = n_points
            
        else:
            self.binded_to_surface_mesh = False
            self._points = nn.Parameter(points, requires_grad=self.learn_positions).to(nerfmodel.device)
            n_points = len(self._points)
        
        # KNN information for training regularization
        self.keep_track_of_knn = keep_track_of_knn
        if keep_track_of_knn:
            self.knn_to_track = knn_to_track
            knns = knn_points(points[None], points[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]
        
        # ---Tools for future meshing---
        # Primitive polygon that will be used to replace the gaussians
        self.primitive_types = primitive_types
        self._diamond_verts = torch.Tensor(
                [[0., -1., 0.], [0., 0, 1.], 
                [0., 1., 0.], [0., 0., -1.]]
                ).to(nerfmodel.device)
        self._square_verts = torch.Tensor(
                [[0., -1., 1.], [0., 1., 1.], 
                [0., 1., -1.], [0., -1., -1.]]
                ).to(nerfmodel.device)
        if primitive_types == 'diamond':
            self.primitive_verts = self._diamond_verts  # Shape (n_vertices_per_gaussian, 3)
        elif primitive_types == 'square':
            self.primitive_verts = self._square_verts  # Shape (n_vertices_per_gaussian, 3)
        self.primitive_triangles = torch.Tensor(
            [[0, 2, 1], [0, 3, 2]]
            ).to(nerfmodel.device).long()  # Shape (n_triangles_per_gaussian, 3)
        self.primitive_border_edges = torch.Tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]]
            ).to(nerfmodel.device).long()  # Shape (n_edges_per_gaussian, 2)
        self.n_vertices_per_gaussian = len(self.primitive_verts)
        self.n_triangles_per_gaussian = len(self.primitive_triangles)
        self.n_border_edges_per_gaussian = len(self.primitive_border_edges)
        self.triangle_scale = triangle_scale
        
        # Texture info
        self._texture_initialized = False
        self.verts_uv, self.faces_uv = None, None
        
        # Render parameters
        self.image_height = int(nerfmodel.training_cameras.height[0].item())
        self.image_width = int(nerfmodel.training_cameras.width[0].item())
        self.focal_factor = max(nerfmodel.training_cameras.p3d_cameras.K[0, 0, 0].item(),
                                nerfmodel.training_cameras.p3d_cameras.K[0, 1, 1].item())
        
        self.fx = nerfmodel.training_cameras.fx[0].item()
        self.fy = nerfmodel.training_cameras.fy[0].item()
        self.fov_x = focal2fov(self.fx, self.image_width)
        self.fov_y = focal2fov(self.fy, self.image_height)
        self.tanfovx = math.tan(self.fov_x * 0.5)
        self.tanfovy = math.tan(self.fov_y * 0.5)
        
        if self.binded_to_surface_mesh and (not learn_surface_mesh_opacity):
            all_densities = inverse_sigmoid(0.9999 * torch.ones((n_points, 1), dtype=torch.float, device=points.device))
            self.learn_opacities = False
        else:
            all_densities = inverse_sigmoid(0.1 * torch.ones((n_points, 1), dtype=torch.float, device=points.device))
        self.all_densities = nn.Parameter(all_densities, 
                                     requires_grad=self.learn_opacities).to(nerfmodel.device)
        self.return_one_densities = False
        
        self.min_ndc_radius = 2. / min(self.image_height, self.image_width)
        self.max_ndc_radius = 2. * 0.01  # 2. * 0.01
        self.min_radius = None # self.min_ndc_radius / self.focal_factor * 0.005  # 0.005
        self.max_radius = None # self.max_ndc_radius / self.focal_factor * 2.  # 2.
        
        self.radius_dim = 7
        
        # Initialize learnable radiuses
        if not self.binded_to_surface_mesh:
            self.scale_activation = scale_activation
            self.scale_inverse_activation = scale_inverse_activation
            
            if initialize:
                radiuses = _initialize_radiuses_gauss_rasterizer(self,)
                print("Initialized radiuses for 3D Gauss Rasterizer")
                
            else:
                radiuses = torch.rand(1, n_points, self.radius_dim, device=nerfmodel.device)
                self.min_radius = self.min_ndc_radius / self.focal_factor * 0.005 # 0.005
                self.max_radius = self.max_ndc_radius / self.focal_factor * 2. # 2.
                
            # 3D Gaussian parameters
            self._scales = nn.Parameter(
                radiuses[0, ..., 4:],
                requires_grad=self.learn_scales).to(nerfmodel.device)
            self._quaternions = nn.Parameter(
                radiuses[0, ..., :4],
                requires_grad=self.learn_quaternions).to(nerfmodel.device)
        
        else:                        
            self.scale_activation = scale_activation
            self.scale_inverse_activation = scale_inverse_activation
            
            # First gather vertices of all triangles
            faces_verts = self._points[self._surface_mesh_faces]  # n_faces, 3, n_coords
            
            # Then, compute initial scales
            scales = (faces_verts - faces_verts[:, [1, 2, 0]]).norm(dim=-1).min(dim=-1)[0] * self.surface_triangle_circle_radius
            scales = scales.clamp_min(0.0000001).reshape(len(faces_verts), -1, 1).expand(-1, self.n_gaussians_per_surface_triangle, 2).clone().reshape(-1, 2)
            self._scales = nn.Parameter(
                scale_inverse_activation(scales),
                requires_grad=self.learn_surface_mesh_scales).to(nerfmodel.device)
            
            # We actually don't learn quaternions here, but complex numbers to encode a 2D rotation in the triangle's plane
            complex_numbers = torch.zeros(self._n_points, 2).to(nerfmodel.device)
            complex_numbers[:, 0] = 1.
            self._quaternions = nn.Parameter(
                complex_numbers,
                requires_grad=self.learn_surface_mesh_scales).to(nerfmodel.device)
            
            # Reference scaling factor
            if self.editable:
                if use_old_method:
                    self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
                else:
                    self._reference_points = self._points.clone().detach()
                    self._reference_normals = self.surface_mesh.faces_normals_list()[0].clone()
                    self.edited_cache = None
        
        # Initialize color features
        self.sh_levels = sh_levels
        sh_coordinates_dc = RGB2SH(colors).unsqueeze(dim=1)
        self._sh_coordinates_dc = nn.Parameter(
            sh_coordinates_dc.to(self.nerfmodel.device),
            requires_grad=True and (not freeze_gaussians)
        ).to(self.nerfmodel.device)
        
        self._sh_coordinates_rest = nn.Parameter(
            torch.zeros(n_points, sh_levels**2 - 1, 3).to(self.nerfmodel.device),
            requires_grad=True and (not freeze_gaussians)
        ).to(self.nerfmodel.device)
            
        # Beta mode
        self.beta_mode = beta_mode
        if beta_mode == 'learnable':
            with torch.no_grad():
                log_beta = self.scale_activation(self._scales).mean().log().view(1,)
            self._log_beta = torch.nn.Parameter(
                log_beta.to(self.nerfmodel.device),
                ).to(self.nerfmodel.device)
    
    @property
    def device(self):
        return self.nerfmodel.device
    
    @property
    def n_points(self):
        if not self.binded_to_surface_mesh:
            return len(self._points)
        else:
            return self._n_points
    
    @property
    def points(self):
        if not self.binded_to_surface_mesh:
            if (not self.learnable_positions) and self.learnable_shifts:
                return self._points + self.max_shift * 2 * (torch.sigmoid(self.shifts) - 0.5)
            else:
                return self._points
        else:
            # First gather vertices of all triangles
            faces_verts = self._points[self._surface_mesh_faces]  # n_faces, 3, n_coords
            
            # Then compute the points using barycenter coordinates in the surface triangles
            points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
            
            return points.reshape(self._n_points, 3)  # n_faces * n_gaussians_per_face, n_coords
    
    @property
    def strengths(self):
        if self.return_one_densities:
            return torch.ones_like(self.all_densities.view(-1, 1))
        else:
            return torch.sigmoid(self.all_densities.view(-1, 1))
        
    @property
    def sh_coordinates(self):
        return torch.cat([self._sh_coordinates_dc, self._sh_coordinates_rest], dim=1)
    
    @property
    def radiuses(self):
        return torch.cat([self._quaternions, self._scales], dim=-1)[None]
    
    @property
    def scaling(self):
        if not self.binded_to_surface_mesh:
            scales = self.scale_activation(self._scales)
        else:
            plane_scales = self.scale_activation(self._scales)
            if self.editable:
                if use_old_method:
                    # Old method described in the original SuGaR paper
                    faces_verts = self._points[self._surface_mesh_faces]
                    faces_centers = faces_verts.mean(dim=1, keepdim=True)
                    scaling_factor = (faces_verts - faces_centers).norm(dim=-1).mean(dim=-1, keepdim=True) / self.reference_scaling_factor
                    plane_scales = plane_scales * scaling_factor[:, None].expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 1)
                else:
                    # New method with better scaling
                    if (self.edited_cache is not None) and self.edited_cache.shape[-1]==3:
                        scales = self.edited_cache
                        self.edited_cache = None
                    else:
                        quaternions, scales = self.get_edited_quaternions_and_scales()
                        self.edited_cache = quaternions
                    return scales

            scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
                plane_scales,
                ], dim=-1)
        return scales
    
    @property
    def quaternions(self):
        if not self.binded_to_surface_mesh:
            quaternions = self._quaternions
        else:
            if (not self.editable) or (use_old_method):                
                # We compute quaternions to enforce face normals to be the first axis of gaussians
                R_0 = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1)

                # We use the first side of every triangle as the second base axis
                faces_verts = self._points[self._surface_mesh_faces]
                base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)

                # We use the cross product for the last base axis
                base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
                
                # We now apply the learned 2D rotation to the base quaternion
                complex_numbers = torch.nn.functional.normalize(self._quaternions, dim=-1).view(len(self._surface_mesh_faces), self.n_gaussians_per_surface_triangle, 2)
                R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
                R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]

                # We concatenate the three vectors to get the rotation matrix
                R = torch.cat([R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
                            R_1[..., None],
                            R_2[..., None]],
                            dim=-1).view(-1, 3, 3)
                quaternions = matrix_to_quaternion(R)
            else:
                if (self.edited_cache is not None) and self.edited_cache.shape[-1]==4:
                    quaternions = self.edited_cache
                    self.edited_cache = None
                else:
                    quaternions, scales = self.get_edited_quaternions_and_scales()
                    self.edited_cache = scales
            
        return torch.nn.functional.normalize(quaternions, dim=-1)
    
    @property
    def triangle_vertices(self):
        # Apply shift to triangle vertices
        if self.primitive_types == 'diamond':
            self.primitive_verts = self._diamond_verts
        elif self.primitive_types == 'square':
            self.primitive_verts = self._square_verts
        else:
            raise ValueError("Unknown primitive type: {}".format(self.primitive_types))
        triangle_vertices = self.primitive_verts[None]  # Shape: (1, n_vertices_per_gaussian, 3)
        
        # Move canonical, shifted triangles to the local gaussian space
        # We need to permute the scaling axes so that the smallest is the first
        scale_argsort = self.scaling.argsort(dim=-1)
        scale_argsort[..., 1] = (scale_argsort[..., 0] + 1) % 3
        scale_argsort[..., 2] = (scale_argsort[..., 0] + 2) % 3
        
        # TODO: Change for a lighter computation that does not require to compute the rotation matrices.
        # We can just permute the axes of triangle_vertices with the inverse permutation.
        
        # Permute scales
        scale_sort = self.scaling.gather(dim=1, index=scale_argsort)
        
        # Permute rotation axes
        rotation_matrices = quaternion_to_matrix(self.quaternions)
        rotation_sort = rotation_matrices.gather(dim=2, index=scale_argsort[..., None, :].expand(-1, 3, -1))
        quaternion_sort = matrix_to_quaternion(rotation_sort)
        
        triangle_vertices = self.points.unsqueeze(1) + quaternion_apply(
            quaternion_sort.unsqueeze(1),
            triangle_vertices * self.triangle_scale * scale_sort.unsqueeze(1))
        
        triangle_vertices = triangle_vertices.view(-1, 3)  # Shape: (n_pts * n_vertices_per_gaussian, 3)
        return triangle_vertices
    
    @property
    def triangle_border_edges(self):
        edges = self.primitive_border_edges[None]  # Shape: (1, n_border_edges_per_gaussian, 2)
        edges = edges + 4 * torch.arange(len(self.points), device=self.device)[:, None, None]  # Shape: (n_pts, n_border_edges_per_gaussian, 2)
        edges = edges.view(-1, 2)  # Shape: (n_pts * n_border_edges_per_gaussian, 2)
        return edges
    
    @property
    def triangles(self):
        triangles = self.primitive_triangles[None].expand(self.n_points, -1, -1).clone()  # Shape: (n_pts, n_triangles_per_gaussian, 3)
        triangles = triangles + 4 * torch.arange(len(self.points), device=self.device)[:, None, None]  # Shape: (n_pts, n_triangles_per_gaussian, 3)
        triangles = triangles.view(-1, 3)  # Shape: (n_pts * n_triangles_per_gaussian, 3)
        return triangles
    
    @property
    def texture_features(self):
        if not self._texture_initialized:
            self.update_texture_features()
        return self.sh_coordinates[self.point_idx_per_pixel]
    
    @property
    def mesh(self):        
        textures_uv = TexturesUV(
            maps=SH2RGB(self.texture_features[..., 0, :][None]), #texture_img[None]),
            verts_uvs=self.verts_uv[None],
            faces_uvs=self.faces_uv[None],
            sampling_mode='nearest',
            )
        
        return Meshes(
            verts=[self.triangle_vertices],   
            faces=[self.triangles],
            textures=textures_uv,
        )
        
    @property
    def surface_mesh(self):
        # Create a Meshes object
        surface_mesh = Meshes(
            verts=[self._points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
            )
        return surface_mesh
    
    def make_editable(self):
        if self.binded_to_surface_mesh and (not self.editable):
            self.editable = True            
            if use_old_method:
                faces_verts = self._points[self._surface_mesh_faces]  # n_faces, 3, n_coords
                self.reference_scaling_factor = (faces_verts - faces_verts.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=-1, keepdim=True)
            else:
                self._reference_points = self._points.clone().detach()
                self._reference_normals = self.surface_mesh.faces_normals_list()[0].clone()
                self.edited_cache = None
            
    def get_edited_quaternions_and_scales(self, verbose=False):
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        
        reference_verts = self._reference_points[self._surface_mesh_faces]
        reference_normals = self._reference_normals
        faces_verts = self._points[self._surface_mesh_faces]
        bary_coords = self.surface_triangle_bary_coords#[None]
        faces_normals = self.surface_mesh.faces_normals_list()[0]
        
        # Then compute the points using barycenter coordinates in the surface triangles
        points = faces_verts[:, None] * bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        
        # Compute initial rotation matrices of faces
        R_0 = torch.nn.functional.normalize(faces_normals, dim=-1)
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        # =====Adjust rotation to the current deformation=====
        reference_base = torch.nn.functional.normalize(reference_verts[:, 0:1] - reference_verts[:, 1:2], dim=-1)  # n_faces, 1, 3
        reference_axis = torch.nn.functional.normalize(reference_verts - reference_verts.mean(dim=-2, keepdim=True), dim=-1)  # n_faces, 3, 3
        reference_axis[:, 2] = -reference_axis[:, 2]
        
        faces_base = torch.nn.functional.normalize(faces_verts[:, 0:1] - faces_verts[:, 1:2], dim=-1)  # n_faces, 1, 3
        faces_axis = torch.nn.functional.normalize(faces_verts - faces_verts.mean(dim=-2, keepdim=True), dim=-1)  # n_faces, 3, 3
        faces_axis[:, 2] = -faces_axis[:, 2]
        
        # Compute angle between the reference and the current sides
        reference_angles = torch.arccos((reference_axis * reference_base).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.))  # n_faces, 3, 1
        faces_angles = torch.arccos((faces_axis * faces_base).sum(dim=-1, keepdim=True).clamp(min=-1., max=1.))  # n_faces, 3, 1
        angles = faces_angles - reference_angles  # n_faces, 3, 1
        point_angles = (angles[:, None] * bary_coords[None]).sum(dim=-2)  # n_faces, n_gaussians_per_face, 1

        # Compute the complex number that will adjust the rotation the gaussians
        point_adjust_complex = torch.cat([torch.cos(point_angles), torch.sin(point_angles)], dim=-1)
        
        # We compute the complex number representing the learned 2D rotation
        complex_numbers = torch.nn.functional.normalize(self._quaternions, dim=-1).view(len(self._surface_mesh_faces), self.n_gaussians_per_surface_triangle, 2)
        
        # We apply the adjustment to the complex numbers
        x, y = complex_numbers[..., 0].clone(), complex_numbers[..., 1].clone()
        a, b = point_adjust_complex[..., 0], point_adjust_complex[..., 1]
        complex_numbers[..., 0] = x * a - y * b
        complex_numbers[..., 1] = x * b + y * a
        
        # We now apply the 2D rotation to the base quaternion
        R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]  # n_faces, n_gaussians_per_face, 3
        R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]  # n_faces, n_gaussians_per_face, 3

        # We concatenate the three vectors to get the rotation matrix, and compute the final quaternion
        R = torch.cat([
            R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_surface_triangle, -1, -1).clone(),
            R_1[..., None],
            R_2[..., None]
            ],
            dim=-1)
        quaternions = matrix_to_quaternion(R)
        
        # =====Adjust scales to the current deformation=====
        all_faces_axis = faces_verts.mean(dim=-2, keepdim=True) - faces_verts  # Shape (n_faces, 3, 3)
        all_faces_axis_norm = torch.norm(all_faces_axis, dim=-1, keepdim=True)  # Shape (n_faces, 3, 1)
        all_faces_axis = torch_normalize(all_faces_axis, dim=-1)  # Shape (n_faces, 3, 3)
        all_faces_orthos = torch.cross(all_faces_axis, faces_normals[:, None], dim=-1)  # Shape (n_faces, 3, 3)
        
        all_reference_axis = reference_verts.mean(dim=-2, keepdim=True) - reference_verts  # Shape (n_faces, 3, 3)
        all_reference_axis_norm = torch.norm(all_reference_axis, dim=-1, keepdim=True)  # Shape (n_faces, 3, 1)
        all_reference_axis = torch_normalize(all_reference_axis, dim=-1)  # Shape (n_faces, 3, 3)
        
        axis_proj_1 = (R_1[..., None, :] * all_faces_axis[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        ortho_proj_1 = (R_1[..., None, :] * all_faces_orthos[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        side_proj_2 = (R_2[..., None, :] * all_faces_axis[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        ortho_proj_2 = (R_2[..., None, :] * all_faces_orthos[:, None]).sum(dim=-1, keepdim=True)  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        
        scaling_1 = torch.sqrt(  (axis_proj_1 * all_faces_axis_norm[:, None] / all_reference_axis_norm[:, None])**2 + ortho_proj_1**2  )  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        scaling_2 = torch.sqrt(  (side_proj_2 * all_faces_axis_norm[:, None] / all_reference_axis_norm[:, None])**2 + ortho_proj_2**2  )  # Shape (n_faces, n_gaussians_per_face, 3, 1)
        
        scaling_1 = (scaling_1 * bary_coords[None]).sum(dim=-2)  # Shape (n_faces, n_gaussians_per_face, 1)
        scaling_2 = (scaling_2 * bary_coords[None]).sum(dim=-2)  # Shape (n_faces, n_gaussians_per_face, 1)
        
        plane_scales = self.scale_activation(self._scales).view(-1, self.n_gaussians_per_surface_triangle, 2)
        plane_scales[..., 0:1] = plane_scales[..., 0:1] * scaling_1
        plane_scales[..., 1:2] = plane_scales[..., 1:2] * scaling_2
        scales = torch.cat([
            self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
            plane_scales.view(-1, 2),
            ], dim=-1)
        
        if verbose:
            with torch.no_grad():
                print("Computed edited quaternions and scales.")
                print("Mean - std - min - max - median")
                print("Scaling1:", scaling_1.mean().item(), scaling_1.std().item(), scaling_1.min().item(), scaling_1.max().item(), scaling_1.median().item())
                print("Scaling2:", scaling_2.mean().item(), scaling_2.std().item(), scaling_2.min().item(), scaling_2.max().item(), scaling_2.median().item())
        return quaternions, scales
    
    def unbind_surface_mesh(self):
        self._quaternions = nn.Parameter(self.quaternions.detach(), requires_grad=self.learn_quaternions).to(self.nerfmodel.device)
        self._scales = nn.Parameter(scale_inverse_activation(self.scaling.detach()), requires_grad=self.learn_scales).to(self.nerfmodel.device)
        self.scale_activation = scale_activation
        self.scale_inverse_activation = scale_inverse_activation
        self.binded_to_surface_mesh = False
            
    def get_filtered_mesh(self, gaussian_mask):        
        new_verts = self.triangle_vertices.clone()#.reshape(-1, self.n_vertices_per_gaussian, 3)
        verts_mask = gaussian_mask[:, None].expand(-1, self.n_vertices_per_gaussian).reshape(-1)
        new_verts[~verts_mask] = self.points.abs().max() * 5.
        new_verts = new_verts#.reshape(-1, 3)
        
        textures_uv = TexturesUV(
            maps=SH2RGB(self.texture_features[..., 0, :][None]), #texture_img[None]),
            verts_uvs=self.verts_uv[None],
            faces_uvs=self.faces_uv[None],
            sampling_mode='nearest',
            )
        
        return Meshes(
            verts=[new_verts],   
            faces=[self.triangles],
            textures=textures_uv,
        )
        
    def splat_mesh(self, p3d_camera, 
                   mode='perspective'  # 'depth' or 'perspective'
                   ):
        new_verts = self.triangle_vertices.clone().reshape(-1, self.n_vertices_per_gaussian, 3)  # Shape (n_points, n_vertices_per_gaussian, 3)
        camera_center = p3d_camera.get_camera_center().view(1, 1, 3)  # Shape (1, 1, 3)
        gaussian_centers = self.points.view(-1, 1, 3)  # Shape (n_points, 1, 3)
        
        n_gaussians = len(gaussian_centers)
        
        new_verts_in_camera_space = p3d_camera.get_world_to_view_transform().transform_points(new_verts.view(-1, 3)).reshape(n_gaussians, -1, 3)  # Shape (n_points, n_vertices_per_gaussian, 3)
        gaussian_centers_in_camera_space = p3d_camera.get_world_to_view_transform().transform_points(gaussian_centers.view(-1, 3)).reshape(n_gaussians, -1, 3)  # Shape (n_points, 1, 3)
        
        if mode == 'depth':
            new_verts_in_camera_space[..., 2] = gaussian_centers_in_camera_space[..., 2]
        else:
            proj_dir = torch.nn.functional.normalize(gaussian_centers_in_camera_space, dim=-1)  # Shape (n_points, 1, 3)
            verts_projection = (new_verts_in_camera_space * proj_dir).sum(-1, keepdim=True)  # Shape (n_points, n_vertices_per_gaussian, 1)
            centers_projection = (gaussian_centers_in_camera_space * proj_dir).sum(-1, keepdim=True)  # Shape (n_points, 1, 1)
            new_verts_in_camera_space = (centers_projection / verts_projection) * new_verts_in_camera_space
        
        splatted_verts = p3d_camera.get_world_to_view_transform().inverse().transform_points(new_verts_in_camera_space.reshape(-1, 3))
        
        textures_uv = TexturesUV(
            maps=SH2RGB(self.texture_features[..., 0, :][None]), #texture_img[None]),
            verts_uvs=self.verts_uv[None],
            faces_uvs=self.faces_uv[None],
            sampling_mode='nearest',
            )
        
        return Meshes(
            verts=[splatted_verts],   
            faces=[self.triangles],
            textures=textures_uv,
        )
        
    def get_covariance(self, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
        scaling = self.scaling
        if inverse_scales:
            scaling = 1. / scaling.clamp(min=1e-8)
        scaled_rotation = quaternion_to_matrix(self.quaternions) * scaling[:, None]
        if return_sqrt:
            return scaled_rotation
        
        cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
        if return_full_matrix:
            return cov3Dmatrix
        
        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)
        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
        
        return cov3D
        
    def update_texture_features(self, square_size_in_texture=2):
        features = self.sh_coordinates.view(len(self.points), -1)
        faces_uv, verts_uv, texture_img, point_idx_per_pixel = _convert_vertex_colors_to_texture(
            self, 
            features,
            square_size=square_size_in_texture,
            )
        self.texture_size = texture_img.shape[0]
        self.verts_uv = verts_uv
        self.faces_uv = faces_uv
        texture_img = texture_img.view(self.texture_size, self.texture_size, -1, 3)

        self.point_idx_per_pixel = torch.nn.Parameter(point_idx_per_pixel, requires_grad=False)
        self._texture_initialized = True
        
    def get_texture_img(self, nerf_cameras, cam_idx, sh_levels:int=None,):
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        if sh_levels is None:
            sh_levels = self.sh_levels
        
        cameras = nerf_cameras.p3d_cameras[cam_idx]
        
        # Compute directions
        directions = torch.nn.functional.normalize(
            self.points[self.point_idx_per_pixel].reshape(-1, 3) - cameras.get_camera_center().view(1, 3), 
            dim=-1)
        
        # Gather sh coordinates
        sh_coordinates = self.texture_features.reshape(len(directions), -1, 3)[:, :sh_levels**2, :]
        
        # Compute texture from sh coordinates
        shs_view = sh_coordinates.transpose(-1, -2).view(-1, 3, sh_levels**2)
        sh2rgb = eval_sh(sh_levels-1, shs_view, directions)
        directional_texture = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
        directional_texture = directional_texture.view(self.texture_size, self.texture_size, 3)
            
        return directional_texture
    
    def prune_points(self, prune_mask):
        print("WARNING! During optimization, you should use a densifier to prune low opacity points.")
        print("This function does not preserve the state of an optimizer, and sets requires_grad=False to all parameters.")
        self._points = torch.nn.Parameter(self._points[prune_mask].detach(), requires_grad=False)
        self._scales = torch.nn.Parameter(self._scales[prune_mask].detach(), requires_grad=False)
        self._quaternions = torch.nn.Parameter(self._quaternions[prune_mask].detach(), requires_grad=False)
        self._sh_coordinates_dc = torch.nn.Parameter(self._sh_coordinates_dc[prune_mask].detach(), requires_grad=False)
        self._sh_coordinates_rest = torch.nn.Parameter(self._sh_coordinates_rest[prune_mask].detach(), requires_grad=False)
        self.all_densities = torch.nn.Parameter(self.all_densities[prune_mask].detach(), requires_grad=False)
        
    def drop_low_opacity_points(self, opacity_threshold=0.5):
        mask = self.strengths[..., 0] > opacity_threshold  # 1e-3, 0.5
        self.prune_points(mask)
        
    def forward(self, **kwargs):
        pass
    
    def adapt_to_cameras(self, cameras:CamerasWrapper):
        self.focal_factor = max(cameras.p3d_cameras.K[0, 0, 0].item(),
                                cameras.p3d_cameras.K[0, 1, 1].item())
        
        self.image_height = int(cameras.height[0].item())
        self.image_width = int(cameras.width[0].item())
        
        self.min_ndc_radius = 2. / min(self.image_height, self.image_width)
        self.max_ndc_radius = 2. * 0.01
        
        self.fx = cameras.fx[0].item()
        self.fy = cameras.fy[0].item()
        self.fov_x = focal2fov(self.fx, self.image_width)
        self.fov_y = focal2fov(self.fy, self.image_height)
        self.tanfovx = math.tan(self.fov_x * 0.5)
        self.tanfovy = math.tan(self.fov_y * 0.5)
        
    def get_cameras_spatial_extent(self, nerf_cameras:CamerasWrapper=None, return_average_xyz=False):
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        
        camera_centers = nerf_cameras.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)  # Should it be replaced by the center of camera bbox, i.e. (min + max) / 2?
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        if return_average_xyz:
            return radius, avg_camera_center
        else:
            return radius
        
    def get_points_rgb(
        self,
        positions:torch.Tensor=None,
        camera_centers:torch.Tensor=None,
        directions:torch.Tensor=None,
        sh_levels:int=None,
        sh_coordinates:torch.Tensor=None,
        ):
        """Returns the RGB color of the points for the given camera pose.

        Args:
            positions (torch.Tensor, optional): Shape (n_pts, 3). Defaults to None.
            camera_centers (torch.Tensor, optional): Shape (n_pts, 3) or (1, 3). Defaults to None.
            directions (torch.Tensor, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
            
        if positions is None:
            positions = self.points

        if camera_centers is not None:
            render_directions = torch.nn.functional.normalize(positions - camera_centers, dim=-1)
        elif directions is not None:
            render_directions = directions
        else:
            raise ValueError("Either camera_centers or directions must be provided.")

        if sh_coordinates is None:
            sh_coordinates = self.sh_coordinates
            
        if sh_levels is None:
            sh_coordinates = sh_coordinates
        else:
            sh_coordinates = sh_coordinates[:, :sh_levels**2]

        shs_view = sh_coordinates.transpose(-1, -2).view(-1, 3, sh_levels**2)
        sh2rgb = eval_sh(sh_levels-1, shs_view, render_directions)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
        
        return colors
    
    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True,):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if mask is None:
            scaling = self.scaling
        else:
            scaling = self.scaling[mask]
        
        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0])
        
        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs()
        # cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)
        cum_probs = areas / areas.sum(dim=-1, keepdim=True)
        
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True)
        if mask is not None:
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]
        
        random_points = self.points[random_indices] + quaternion_apply(
            self.quaternions[random_indices], 
            sampling_scale_factor * self.scaling[random_indices] * torch.randn_like(self.points[random_indices]))
        
        return random_points, random_indices
    
    def get_smallest_axis(self, return_idx=False):
        """Returns the smallest axis of the Gaussians.

        Args:
            return_idx (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rotation_matrices = quaternion_to_matrix(self.quaternions)
        smallest_axis_idx = self.scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normals(self, estimate_from_points=False, neighborhood_size:int=32):
        """Returns the normals of the Gaussians.

        Args:
            estimate_from_points (bool, optional): _description_. Defaults to False.
            neighborhood_size (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
        """
        if estimate_from_points:
            normals = estimate_pointcloud_normals(
                self.points[None], #.detach(), 
                neighborhood_size=neighborhood_size,
                disambiguate_directions=True
                )[0]
        else:
            if self.binded_to_surface_mesh:
                normals = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1).view(-1, 1, 3)
                normals = normals.expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 3)
            else:
                normals = self.get_smallest_axis()
        return normals
    
    def get_neighbors_of_random_points(self, num_samples):
        if num_samples >= 0:
            sampleidx = torch.randperm(len(self.points), device=self.device)[:num_samples]        
            return self.knn_idx[sampleidx]
        else:
            return self.knn_idx
    
    def get_local_variance(self, values:torch.Tensor, neighbor_idx:torch.Tensor):
        """_summary_

        Args:
            values (_type_): Shape is (n_points, n_values)
            neighbor_idx (_type_): Shape is (n_points, n_neighbors)
        """
        neighbor_values = values[neighbor_idx]  # Shape is (n_points, n_neighbors, n_values)
        return (neighbor_values - neighbor_values.mean(dim=1, keepdim=True)).pow(2).sum(dim=-1).mean(dim=1)
    
    def get_local_distance2(
        self, 
        values:torch.Tensor, 
        neighbor_idx:torch.Tensor, 
        weights:torch.Tensor=None,
        ):
        """_summary_

        Args:
            values (torch.Tensor): Shape is (n_points, n_values)
            neighbor_idx (torch.Tensor): Shape is (n_points, n_neighbors)
            weights (torch.Tensor, optional): Shape is (n_points, n_neighbors). Defaults to None.

        Returns:
            _type_: _description_
        """
        
        neighbor_values = values[neighbor_idx]  # Shape is (n_points, n_neighbors, n_values)
        distance2 = neighbor_values[:, 1:] - neighbor_values[:, :1]  # Shape is (n_points, n_neighbors-1, n_values)
        distance2 = distance2.pow(2).sum(dim=-1)  # Shape is (n_points, n_neighbors-1)
        
        if weights is not None:
            distance2 = distance2 * weights

        return distance2.mean(dim=1)  # Shape is (n_points,)
    
    def reset_neighbors(self, knn_to_track:int=None):
        if self.binded_to_surface_mesh:
            print("WARNING! You should not reset the neighbors of a surface mesh.")
            print("Then, neighbors reset will be ignored.")
        else:
            if not hasattr(self, 'knn_to_track'):
                if knn_to_track is None:
                    knn_to_track = 16
                self.knn_to_track = knn_to_track
            else:
                if knn_to_track is None:
                    knn_to_track = self.knn_to_track 
            # Compute KNN               
            with torch.no_grad():
                self.knn_to_track = knn_to_track
                knns = knn_points(self.points[None], self.points[None], K=knn_to_track)
                self.knn_dists = knns.dists[0]
                self.knn_idx = knns.idx[0]
            
    def get_edge_neighbors(self, k_neighbors, 
                           edges=None, triangle_vertices=None,):
        if edges is None:
            edges = self.triangle_border_edges
        if triangle_vertices is None:
            triangle_vertices = self.triangle_vertices
        
        # We select the closest edges based on the position of the edge center
        edge_centers = triangle_vertices[edges].mean(dim=-2)
        
        # TODO: Compute only for vertices with high opacity? Remove points with low opacity?
        edge_knn = knn_points(edge_centers[None], edge_centers[None], K=8)
        edge_knn_idx = edge_knn.idx[0]
        
        return edge_knn_idx
            
    def compute_gaussian_overlap_with_neighbors(
        self, 
        neighbor_idx,
        use_gaussian_center_only=True,
        n_samples_to_compute_overlap=32,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        ):
        
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:, 0]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self.scaling.detach()
            quaternions = self.quaternions.detach()
        else:
            scaling = self.scaling
            quaternions = self.quaternions
        
        # Samples points in the corresponding gaussians
        if use_gaussian_center_only:
            n_samples_to_compute_overlap = 1
            gaussian_samples = self.points[point_idx].unsqueeze(1) + 0.  # (n_points, n_samples_to_compute_overlap, 3)
        else:
            gaussian_samples = self.points[point_idx].unsqueeze(1) + quaternion_apply(
                quaternions[point_idx].unsqueeze(1), 
                scaling[point_idx].unsqueeze(1) * torch.randn(
                    n_points, n_samples_to_compute_overlap, 3, 
                    device=self.device)
                )  # (n_points, n_samples_to_compute_overlap, 3)
        
        # >>> We will now compute the gaussian weight of all samples, for each neighbor gaussian.
        # We start by computing the shift between the samples and the neighbor gaussian centers.
        neighbor_center_to_samples = gaussian_samples.unsqueeze(1) - self.points[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
        # We compute the inverse of the scaling of the neighbor gaussians. 
        # For 2D gaussians, we implictly project the samples on the plane of each gaussian; 
        # We do so by setting the inverse of the scaling of the gaussian to 0 in the direction of the gaussian normal (i.e. 0-axis).
        inverse_scales = 1. / scaling[neighbor_idx[:, neighbor_start_idx:]].unsqueeze(2)  # (n_points, n_neighbors-1, 1, 3)
        
        # We compute the "gaussian distance" of all samples to the neighbor gaussians, i.e. the norm of the unrotated shift,
        # weighted by the inverse of the scaling of the neighbor gaussians.
        gaussian_distances = inverse_scales * quaternion_apply(
            quaternion_invert(quaternions[neighbor_idx[:, neighbor_start_idx:]]).unsqueeze(2), 
            neighbor_center_to_samples
            )  # (n_points, n_neighbors-1, n_samples_to_compute_overlap, 3)
        
        # Now we can compute the gaussian weights of all samples, for each neighbor gaussian.
        # We then sum them to get the gaussian overlap of each neighbor gaussian.
        gaussian_weights = torch.exp(-1./2. * (gaussian_distances ** 2).sum(dim=-1))  # (n_points, n_neighbors-1, n_samples_to_compute_overlap)
        gaussian_overlaps = gaussian_weights.mean(dim=-1)  # (n_points, n_neighbors-1)
        
        # If needed, we weight the gaussian overlaps by the angle between the normal of the neighbor gaussian and the normal of the point gaussian
        if weight_by_normal_angle:
            normals = self.get_normals()[neighbor_idx]  # (n_points, n_neighbors, 3)
            weights = (normals[:, 1:] * normals[:, 0:1]).sum(dim=-1).abs()  # (n_points, n_neighbors-1)
            gaussian_overlaps = gaussian_overlaps * weights
            
        return gaussian_overlaps
    
    def compute_gaussian_alignment_with_neighbors(
        self,
        neighbor_idx,
        weight_by_normal_angle=False,
        propagate_gradient_to_points_only=False,
        std_factor = 1.,
        ):
        
        # This is used to skip the first neighbor, which is the point itself
        neighbor_start_idx = 1
        
        # Get sampled points
        point_idx = neighbor_idx[:, 0]  # (n_points, )
        n_points = len(point_idx)
        
        # Decide whether we want to propagate the gradient to the points only, or to the points and the covariance parameters
        if propagate_gradient_to_points_only:
            scaling = self.scaling.detach()
            quaternions = self.quaternions.detach()
        else:
            scaling = self.scaling
            quaternions = self.quaternions
        
        # We compute scaling, inverse quaternions and centers for all gaussians and their neighbors
        all_scaling = scaling[neighbor_idx]
        all_invert_quaternions = quaternion_invert(quaternions)[neighbor_idx]
        all_centers = self.points[neighbor_idx]
        
        # We compute direction vectors between the gaussians and their neighbors
        neighbor_shifts = all_centers[:, neighbor_start_idx:] - all_centers[:, :neighbor_start_idx]
        neighbor_distances = neighbor_shifts.norm(dim=-1).clamp(min=1e-8)
        neighbor_directions = neighbor_shifts / neighbor_distances.unsqueeze(-1)
        
        # We compute the standard deviations of the gaussians in the direction of their neighbors,
        # and reciprocally in the direction of the gaussians.
        standard_deviations_gaussians = (
            all_scaling[:, 0:neighbor_start_idx]
            * quaternion_apply(all_invert_quaternions[:, 0:neighbor_start_idx], 
                               neighbor_directions)
            ).norm(dim=-1)
        
        standard_deviations_neighbors = (
            all_scaling[:, neighbor_start_idx:]
            * quaternion_apply(all_invert_quaternions[:, neighbor_start_idx:], 
                               neighbor_directions)
            ).norm(dim=-1)
        
        # The distance between the gaussians and their neighbors should be the sum of their standard deviations (up to a factor)
        stabilized_distance = (standard_deviations_gaussians + standard_deviations_neighbors) * std_factor
        gaussian_alignment = (neighbor_distances / stabilized_distance.clamp(min=1e-8) - 1.).abs()
        
        # If needed, we weight the gaussian alignments by the angle between the normal of the neighbor gaussian and the normal of the point gaussian
        if weight_by_normal_angle:
            normals = self.get_normals()[neighbor_idx]  # (n_points, n_neighbors, 3)
            weights = (normals[:, 1:] * normals[:, 0:1]).sum(dim=-1).abs()  # (n_points, n_neighbors-1)
            gaussian_alignment = gaussian_alignment * weights
            
        return gaussian_alignment
    
    def get_beta(self, x, 
                 closest_gaussians_idx=None, 
                 closest_gaussians_opacities=None,
                 densities=None,
                 opacity_min_clamp=1e-32,):
        """_summary_

        Args:
            x (_type_): Should have shape (n_points, 3)
            closest_gaussians_idx (_type_, optional): Should have shape (n_points, n_neighbors).
                Defaults to None.
            closest_gaussians_opacities (_type_, optional): Should have shape (n_points, n_neighbors).
            densities (_type_, optional): Should have shape (n_points, ).

        Returns:
            _type_: _description_
        """
        if self.beta_mode == 'learnable':
            return torch.exp(self._log_beta).expand(len(x))
        
        elif self.beta_mode == 'average':
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='average'.")
            return self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
        
        elif self.beta_mode == 'weighted_average':
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='weighted_average'.")
            if closest_gaussians_opacities is None:
                raise ValueError("closest_gaussians_opacities must be provided when using beta_mode='weighted_average'.")
            
            min_scaling = self.scaling.min(dim=-1)[0][closest_gaussians_idx]
            
            # if densities is None:
            if True:
                opacities_sum = closest_gaussians_opacities.sum(dim=-1, keepdim=True)
            else:
                opacities_sum = densities.view(-1, 1)
            # weights = neighbor_opacities.clamp(min=opacity_min_clamp) / opacities_sum.clamp(min=opacity_min_clamp)
            weights = closest_gaussians_opacities / opacities_sum.clamp(min=opacity_min_clamp)

            # Three methods to handle the case where all opacities are 0.
            # Important because we need to avoid beta == 0 at all cost for these points!
            # Indeed, beta == 0. gives sdf == 0.
            # However these points are far from gaussians, so they should have a sdf != 0.

            # Method 1: Give 1-weight to closest gaussian (Not good)
            if False:
                one_at_closest_gaussian = torch.zeros(1, neighbor_opacities.shape[1], device=rc.device)
                one_at_closest_gaussian[0, 0] = 1.
                weights[opacities_sum[..., 0] == 0.] = one_at_closest_gaussian
                beta = (rc.scaling.min(dim=-1)[0][closest_gaussians_idx] * weights).sum(dim=1)
            
            # Method 2: Give the maximum scaling value in neighbors as beta (Not good if neighbors have low scaling)
            if False:
                beta = (min_scaling * weights).sum(dim=-1)
                mask = opacities_sum[..., 0] == 0.
                beta[mask] = min_scaling.max(dim=-1)[0][mask]
            
            # Method 3: Give a constant, large beta value (better control)
            if True:
                beta = (min_scaling * weights).sum(dim=-1)
                with torch.no_grad():
                    if False:
                        # Option 1: beta = camera_spatial_extent
                        beta[opacities_sum[..., 0] == 0.] = rc.get_cameras_spatial_extent()
                    else:
                        # Option 2: beta = largest min_scale in the scene
                        beta[opacities_sum[..., 0] == 0.] = min_scaling.max().detach()
            
            return beta
        
        else:
            raise ValueError("Unknown beta_mode.")
    
    def get_field_values(self, x, gaussian_idx=None, 
                    closest_gaussians_idx=None,
                    gaussian_strengths=None, 
                    gaussian_centers=None, 
                    gaussian_inv_scaled_rotation=None,
                    return_sdf=True, density_threshold=1., density_factor=1.,
                    return_sdf_grad=False, sdf_grad_max_value=10.,
                    opacity_min_clamp=1e-16,
                    return_closest_gaussian_opacities=False,
                    return_beta=False,):
        if gaussian_strengths is None:
            gaussian_strengths = self.strengths
        if gaussian_centers is None:
            gaussian_centers = self.points
        if gaussian_inv_scaled_rotation is None:
            gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True)
        
        if closest_gaussians_idx is None:
            closest_gaussians_idx = self.knn_idx[gaussian_idx]
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx]
        closest_gaussian_strengths = gaussian_strengths[closest_gaussians_idx]
        
        fields = {}
        
        # Compute the density field as a sum of local gaussian opacities
        # TODO: Change the normalization of the density (maybe learn the scaling parameter?)
        shift = (x[:, None] - closest_gaussian_centers)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        fields['density'] = densities.clone()
        density_mask = densities >= 1.
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)
        
        if return_closest_gaussian_opacities:
            fields['closest_gaussian_opacities'] = neighbor_opacities
        
        if return_sdf or return_sdf_grad or return_beta:
            # --- Old way
            # beta = self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
            # ---New way
            beta = self.get_beta(x, 
                                 closest_gaussians_idx=closest_gaussians_idx, 
                                 closest_gaussians_opacities=neighbor_opacities, 
                                 densities=densities,
                                 opacity_min_clamp=opacity_min_clamp,
                                 )
            clamped_densities = densities.clamp(min=opacity_min_clamp)

        if return_beta:
            fields['beta'] = beta
        
        # Compute the signed distance field
        if return_sdf:
            sdf_values = beta * (
                torch.sqrt(-2. * torch.log(clamped_densities)) # TODO: Change the max=1. to something else?
                - np.sqrt(-2. * np.log(min(density_threshold, 1.)))
                )
            fields['sdf'] = sdf_values
            
        # Compute the gradient of the signed distance field
        if return_sdf_grad:
            sdf_grad = neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]
            sdf_grad = sdf_grad.sum(dim=-2)
            sdf_grad = (beta / (clamped_densities * torch.sqrt(-2. * torch.log(clamped_densities))).clamp(min=opacity_min_clamp))[..., None] * sdf_grad
            fields['sdf_grad'] = sdf_grad.clamp(min=-sdf_grad_max_value, max=sdf_grad_max_value)
            
        return fields
    
    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space):
        depth_view = depth.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
        pts_projections = fov_camera.get_projection_transform().transform_points(points_in_camera_space)

        factor = -1 * min(self.image_height, self.image_width)
        # todo: Parallelize these two lines with a tensor [image_width, image_height]
        pts_projections[..., 0] = factor / self.image_width * pts_projections[..., 0]
        pts_projections[..., 1] = factor / self.image_height * pts_projections[..., 1]
        pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border'  # 'reflection', 'zeros'
                                                )[0, 0, :, 0]
        return map_z
    
    def get_gaussians_closest_to_samples(self, x, n_closest_gaussian=None):
        if n_closest_gaussian is None:
            if not hasattr(self, 'knn_to_track'):
                print("Variable knn_to_track not found. Setting it to 16.")
                self.knn_to_track = 16
            n_closest_gaussian = self.knn_to_track
        
        closest_gaussians_idx = knn_points(x[None], self.points[None], K=n_closest_gaussian).idx[0]
        return closest_gaussians_idx
    
    def compute_density(self, x, closest_gaussians_idx=None, density_factor=1., 
                        return_closest_gaussian_opacities=False):
        
        if closest_gaussians_idx is None:
            closest_gaussians_idx = self.get_gaussians_closest_to_samples(x)
        
        # Gather gaussian parameters
        close_gaussian_centers = self.points[closest_gaussians_idx]
        close_gaussian_inv_scaled_rotation = self.get_covariance(
            return_full_matrix=True, return_sqrt=True, inverse_scales=True
            )[closest_gaussians_idx]
        close_gaussian_strengths = self.strengths[closest_gaussians_idx]
        
        # Compute the density field as a sum of local gaussian opacities
        shift = (x[:, None] - close_gaussian_centers)
        warped_shift = close_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = density_factor * close_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        
        if return_closest_gaussian_opacities:
            return densities, neighbor_opacities
        else:
            return densities  # Shape is (n_points, )
        
    def get_signed_normals(self, gaussian_idx, gaussian_sign_encodings):
        """_summary_

        Args:
            gaussian_idx (_type_): Should have shape (n_points, )
            gaussian_sign_encodings (_type_): Should have shape (n_points, n_sh, 1)

        Returns:
            _type_: _description_
        """
        n_sh = gaussian_sign_encodings.shape[-2]
        
        normals = self.get_normals()[gaussian_idx]
        quaternions = self.quaternions[gaussian_idx]
        normal_signs = 2. * eval_sh(
            deg=self.gaussian_pos_encoding_cfg.encodings_sh_deg,
            sh=gaussian_sign_encodings.transpose(-1, -2).view(-1, 1, n_sh),
            dirs=quaternion_apply(quaternion_invert(quaternions), normals).view(-1, 3)
        )
        
        return torch.nn.functional.normalize(normal_signs * normals, dim=-1)
    
    def compute_level_surface_points_from_camera(
        self, 
        nerf_cameras=None,
        cam_idx=0,
        rasterizer=None,
        surface_level:float=0.5,  # 0.5 or 0.1
        n_surface_points=-1,
        primitive_types=None,  # Should be 'diamond'
        triangle_scale=None,  # Should be 2.
        splat_mesh=True,  # True
        n_points_in_range=21,  # 21
        range_size=3.,  # 3.
        n_points_per_pass=2_000_000,
        density_factor=1.,
        return_depth=False,
        return_gaussian_idx=False,
        return_normals=False,
        compute_flat_normals=False,
        compute_intersection_for_flat_gaussian=False,  # Should be False
        use_gaussian_depth=False,  # False until now. TODO: Test with True
        just_use_depth_as_level=False,  # Should be False
        ):
        # Remember to reset neighbors and update texture features before calling this function
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        
        if primitive_types is not None:
            self.primitive_types = primitive_types
            
        if triangle_scale is not None:
            self.triangle_scale = triangle_scale
            
        if rasterizer is None:
            faces_per_pixel = 10
            max_faces_per_bin = 50_000

            mesh_raster_settings = RasterizationSettings(
                image_size=(self.image_height, self.image_width),
                blur_radius=0.0, 
                faces_per_pixel=faces_per_pixel,
                max_faces_per_bin=max_faces_per_bin
            )
            rasterizer = MeshRasterizer(
                    cameras=nerf_cameras.p3d_cameras[cam_idx], 
                    raster_settings=mesh_raster_settings,
                )
            
        p3d_cameras = nerf_cameras.p3d_cameras[cam_idx]
        
        # Compute splatted depth
        if use_gaussian_depth:
            point_depth = p3d_cameras.get_world_to_view_transform().transform_points(self.points)[..., 2:].expand(-1, 3)
            depth = self.render_image_gaussian_rasterizer( 
                    camera_indices=cam_idx,
                    bg_color=torch.Tensor([-1., -1., -1.]).to(self.device),
                    sh_deg=0,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=False,
                    use_same_scale_in_all_directions=False,
                    point_colors=point_depth,
                ).contiguous()[..., 0]
            no_depth_mask = depth < 0.
            depth[no_depth_mask] = depth.max() * 1.05
        else:
            if True:
                textures_img = self.get_texture_img(
                    nerf_cameras=nerf_cameras, 
                    cam_idx=cam_idx,
                    sh_levels=self.sh_levels,
                    )

            if splat_mesh:
                mesh = self.splat_mesh(p3d_cameras)
            else:
                mesh = self.mesh
            if True:
                mesh.textures._maps_padded = textures_img[None]

            fragments = rasterizer(mesh, cameras=p3d_cameras)
            depth = fragments.zbuf[0, ..., 0]
            no_depth_mask = depth < 0.
            depth[no_depth_mask] = depth.max() * 1.05
        
        # We backproject the points in world space
        batch_size = 1
        x_tab = torch.Tensor([[i for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        y_tab = torch.Tensor([[j for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        ndc_x_tab = self.image_width / min(self.image_width,
                                                self.image_height) - (y_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2
        ndc_y_tab = self.image_height / min(self.image_width,
                                                    self.image_height) - (x_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2

        ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                ndc_y_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                depth.view(batch_size, -1, 1)),
                                dim=-1
                                ).view(batch_size, self.image_height * self.image_width, 3)
        fov_cameras = nerf_cameras.p3d_cameras[cam_idx]
        no_proj_mask = no_depth_mask.view(-1)
        all_world_points = fov_cameras.unproject_points(ndc_points, scaled_depth_input=False).view(-1, 3)
        
        # Gather info about gaussians surrounding each 3D point
        if use_gaussian_depth:
            closest_gaussians_idx = self.get_gaussians_closest_to_samples(all_world_points)
            gaussian_idx = closest_gaussians_idx[..., 0]
        else:
            gaussian_idx = fragments.pix_to_face[..., 0].view(-1) // self.n_triangles_per_gaussian
            closest_gaussians_idx = self.knn_idx[gaussian_idx]
        
        # We compute the standard deviation of the gaussian at each point
        gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)
        gaussian_standard_deviations = (self.scaling * quaternion_apply(quaternion_invert(self.quaternions), gaussian_to_camera)).norm(dim=-1)
        points_stds = gaussian_standard_deviations[closest_gaussians_idx[..., 0]]
        
        # We compute ray samples
        points_range = torch.linspace(-range_size, range_size, n_points_in_range).to(self.device).view(1, -1, 1)  # (1, n_points_in_range, 1)
        points_range = points_range * points_stds[..., None, None].expand(-1, n_points_in_range, 1)  # (n_points, n_points_in_range, 1)
        camera_to_samples = torch.nn.functional.normalize(all_world_points - fov_cameras.get_camera_center(), dim=-1)  # (n_points, 3)
        samples = (all_world_points[:, None, :] + points_range * camera_to_samples[:, None, :]).view(-1, 3)  # (n_points * n_points_in_range, 3)
        samples_closest_gaussians_idx = closest_gaussians_idx[:, None, :].expand(-1, n_points_in_range, -1).reshape(-1, self.knn_to_track)
        
        # Compute densities of all samples
        densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
        gaussian_strengths = self.strengths
        gaussian_centers = self.points
        gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True)
        
        for i in range(0, len(samples), n_points_per_pass):
            i_start = i
            i_end = min(len(samples), i + n_points_per_pass)
            
            pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]
            
            closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[pass_closest_gaussians_idx]
            closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]

            shift = (samples[i_start:i_end, None] - closest_gaussian_centers)
            if not compute_intersection_for_flat_gaussian:
                warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
            else:
                closest_gaussian_normals = self.get_normals()[pass_closest_gaussians_idx]
                closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][pass_closest_gaussians_idx]
                neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
            neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
            pass_densities = neighbor_opacities.sum(dim=-1)
            pass_density_mask = pass_densities >= 1.
            pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (pass_densities[pass_density_mask].detach() + 1e-12)
            
            densities[i_start:i_end] = pass_densities
        densities = densities.reshape(-1, n_points_in_range)
        
        # Compute first surface intersection points
        if not just_use_depth_as_level:
            under_level = (densities - surface_level < 0)
            above_level = (densities - surface_level > 0)

            _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
            empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

            valid_densities = densities[~empty_pixels]
            valid_range = points_range[~empty_pixels][..., 0]
            valid_first_point_above_level = first_point_above_level[~empty_pixels]

            first_value_above_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level).view(-1)
            value_before_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

            first_t_above_level = valid_range.gather(dim=-1, index=valid_first_point_above_level).view(-1)
            t_before_level = valid_range.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

            intersection_t = (surface_level - value_before_level) / (first_value_above_level - value_before_level) * (first_t_above_level - t_before_level) + t_before_level
            intersection_points = (all_world_points[~empty_pixels] + intersection_t[:, None] * camera_to_samples[~empty_pixels])
        else:
            intersection_points = all_world_points[~no_proj_mask]
            empty_pixels = torch.zeros(densities.shape[0], dtype=torch.bool, device=self.device)
        
        if return_depth or return_gaussian_idx or return_normals:
            outputs = {}
            
            if return_depth:
                valid_z = p3d_cameras.get_world_to_view_transform().transform_points(intersection_points)[..., 2]
                new_zbuf = -torch.ones(self.image_height * self.image_width, dtype=torch.float, device=self.device)
                new_zbuf[~empty_pixels] = valid_z
                new_zbuf = new_zbuf.reshape(self.image_height, self.image_width)
                outputs['new_zbuf'] = new_zbuf
                outputs['empty_pixels'] = empty_pixels
                
            if return_gaussian_idx:
                outputs['gaussian_idx'] = gaussian_idx[~empty_pixels]
            
            if return_normals:                
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]

                closest_gaussian_centers = gaussian_centers[points_closest_gaussians_idx]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[points_closest_gaussians_idx]
                closest_gaussian_strengths = gaussian_strengths[points_closest_gaussians_idx]

                shift = (intersection_points[:, None] - closest_gaussian_centers)
                if not compute_intersection_for_flat_gaussian:
                    warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
                else:
                    closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                    closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][points_closest_gaussians_idx]
                    neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
                neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
                
                if not compute_flat_normals:
                    density_grad = (neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]).sum(dim=-2)
                else:
                    closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                    closest_gaussian_min_scales = self.scaling.min(dim=-1, keepdim=True)[0][points_closest_gaussians_idx]
                    density_grad = (
                        neighbor_opacities[..., None] * 
                        1. / (closest_gaussian_min_scales).pow(2)  * (shift * closest_gaussian_normals).sum(dim=-1, keepdim=True) * closest_gaussian_normals
                        ).sum(dim=-2)
                
                intersection_normals = -torch.nn.functional.normalize(density_grad, dim=-1)
                outputs['normals'] = intersection_normals
                
            return intersection_points, outputs

        return intersection_points
    
    def compute_level_surface_points_from_camera_efficient(
        self, 
        nerf_cameras=None,
        cam_idx=0,
        rasterizer=None,
        surface_levels=[0.1, 0.3, 0.5],  # 0.5 or 0.1
        n_surface_points=-1,
        primitive_types=None,  # Should be 'diamond'
        triangle_scale=None,  # Should be 2.
        splat_mesh=True,  # True
        n_points_in_range=21,  # 21
        range_size=3.,  # 3.
        n_points_per_pass=2_000_000,
        density_factor=1.,
        return_depth=False,
        return_gaussian_idx=False,
        return_normals=False,
        compute_flat_normals=False,
        compute_intersection_for_flat_gaussian=False,  # Should be False
        use_gaussian_depth=False,  # False until now. TODO: Test with True
        just_use_depth_as_level=False, # Should be False
        ):
        # Remember to reset neighbors and update texture features before calling this function
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        
        if primitive_types is not None:
            self.primitive_types = primitive_types
            
        if triangle_scale is not None:
            self.triangle_scale = triangle_scale
            
        if rasterizer is None:
            faces_per_pixel = 10
            max_faces_per_bin = 50_000

            mesh_raster_settings = RasterizationSettings(
                image_size=(self.image_height, self.image_width),
                blur_radius=0.0, 
                faces_per_pixel=faces_per_pixel,
                max_faces_per_bin=max_faces_per_bin
            )
            rasterizer = MeshRasterizer(
                    cameras=nerf_cameras.p3d_cameras[cam_idx], 
                    raster_settings=mesh_raster_settings,
                )
            
        p3d_cameras = nerf_cameras.p3d_cameras[cam_idx]
        
        # Compute splatted depth
        if use_gaussian_depth:
            point_depth = p3d_cameras.get_world_to_view_transform().transform_points(self.points)[..., 2:].expand(-1, 3)
            depth = self.render_image_gaussian_rasterizer( 
                    camera_indices=cam_idx,
                    bg_color=torch.Tensor([-1., -1., -1.]).to(self.device),
                    sh_deg=0,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=False,
                    use_same_scale_in_all_directions=False,
                    point_colors=point_depth,
                ).contiguous()[..., 0]
            no_depth_mask = depth < 0.
            depth[no_depth_mask] = depth.max() * 1.05
        else:
            if True:
                textures_img = self.get_texture_img(
                    nerf_cameras=nerf_cameras, 
                    cam_idx=cam_idx,
                    sh_levels=self.sh_levels,
                    )

            if splat_mesh:
                mesh = self.splat_mesh(p3d_cameras)
            else:
                mesh = self.mesh
            if True:
                mesh.textures._maps_padded = textures_img[None]

            fragments = rasterizer(mesh, cameras=p3d_cameras)
            depth = fragments.zbuf[0, ..., 0]
            no_depth_mask = depth < 0.
            depth[no_depth_mask] = depth.max() * 1.05
        
        # We backproject the points in world space
        batch_size = 1
        x_tab = torch.Tensor([[i for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        y_tab = torch.Tensor([[j for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        ndc_x_tab = self.image_width / min(self.image_width,
                                                self.image_height) - (y_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2
        ndc_y_tab = self.image_height / min(self.image_width,
                                                    self.image_height) - (x_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2

        ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                ndc_y_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                depth.view(batch_size, -1, 1)),
                                dim=-1
                                ).view(batch_size, self.image_height * self.image_width, 3)
        fov_cameras = nerf_cameras.p3d_cameras[cam_idx]
        no_proj_mask = no_depth_mask.view(-1)
        all_world_points = fov_cameras.unproject_points(ndc_points, scaled_depth_input=False).view(-1, 3)
        
        # Gather info about gaussians surrounding each 3D point
        if use_gaussian_depth:
            closest_gaussians_idx = self.get_gaussians_closest_to_samples(all_world_points)
            gaussian_idx = closest_gaussians_idx[..., 0]
        else:
            gaussian_idx = fragments.pix_to_face[..., 0].view(-1) // self.n_triangles_per_gaussian
            closest_gaussians_idx = self.knn_idx[gaussian_idx]
        
        # We compute the standard deviation of the gaussian at each point
        gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)
        gaussian_standard_deviations = (self.scaling * quaternion_apply(quaternion_invert(self.quaternions), gaussian_to_camera)).norm(dim=-1)
        points_stds = gaussian_standard_deviations[closest_gaussians_idx[..., 0]]
        
        # We compute ray samples
        points_range = torch.linspace(-range_size, range_size, n_points_in_range).to(self.device).view(1, -1, 1)  # (1, n_points_in_range, 1)
        points_range = points_range * points_stds[..., None, None].expand(-1, n_points_in_range, 1)  # (n_points, n_points_in_range, 1)
        camera_to_samples = torch.nn.functional.normalize(all_world_points - fov_cameras.get_camera_center(), dim=-1)  # (n_points, 3)
        samples = (all_world_points[:, None, :] + points_range * camera_to_samples[:, None, :]).view(-1, 3)  # (n_points * n_points_in_range, 3)
        samples_closest_gaussians_idx = closest_gaussians_idx[:, None, :].expand(-1, n_points_in_range, -1).reshape(-1, self.knn_to_track)
        
        # Compute densities of all samples
        densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
        gaussian_strengths = self.strengths
        gaussian_centers = self.points
        gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True)
        
        for i in range(0, len(samples), n_points_per_pass):
            i_start = i
            i_end = min(len(samples), i + n_points_per_pass)
            
            pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]
            
            closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[pass_closest_gaussians_idx]
            closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]

            shift = (samples[i_start:i_end, None] - closest_gaussian_centers)
            if not compute_intersection_for_flat_gaussian:
                warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
            else:
                closest_gaussian_normals = self.get_normals()[pass_closest_gaussians_idx]
                closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][pass_closest_gaussians_idx]
                neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
            neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
            pass_densities = neighbor_opacities.sum(dim=-1)
            pass_density_mask = pass_densities >= 1.
            pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (pass_densities[pass_density_mask].detach() + 1e-12)
            
            densities[i_start:i_end] = pass_densities
        densities = densities.reshape(-1, n_points_in_range)
        
        # Compute first surface intersection points
        all_outputs = {}
        for surface_level in surface_levels:
            outputs = {}
                        
            under_level = (densities - surface_level < 0)
            above_level = (densities - surface_level > 0)

            _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
            empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

            if not just_use_depth_as_level:
                valid_densities = densities[~empty_pixels]
                valid_range = points_range[~empty_pixels][..., 0]
                valid_first_point_above_level = first_point_above_level[~empty_pixels]

                first_value_above_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level).view(-1)
                value_before_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

                first_t_above_level = valid_range.gather(dim=-1, index=valid_first_point_above_level).view(-1)
                t_before_level = valid_range.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

                intersection_t = (surface_level - value_before_level) / (first_value_above_level - value_before_level) * (first_t_above_level - t_before_level) + t_before_level
                intersection_points = (all_world_points[~empty_pixels] + intersection_t[:, None] * camera_to_samples[~empty_pixels])
            else:
                empty_pixels = torch.zeros_like(empty_pixels, dtype=torch.bool)
                intersection_points = all_world_points[~empty_pixels]
            outputs['intersection_points'] = intersection_points
            
            if return_depth:
                valid_z = p3d_cameras.get_world_to_view_transform().transform_points(intersection_points)[..., 2]
                new_zbuf = -torch.ones(self.image_height * self.image_width, dtype=torch.float, device=self.device)
                new_zbuf[~empty_pixels] = valid_z
                new_zbuf = new_zbuf.reshape(self.image_height, self.image_width)
                outputs['new_zbuf'] = new_zbuf
                outputs['empty_pixels'] = empty_pixels
                
            if return_gaussian_idx:
                outputs['gaussian_idx'] = gaussian_idx[~empty_pixels]
            
            if return_normals:                
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]

                closest_gaussian_centers = gaussian_centers[points_closest_gaussians_idx]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[points_closest_gaussians_idx]
                closest_gaussian_strengths = gaussian_strengths[points_closest_gaussians_idx]

                shift = (intersection_points[:, None] - closest_gaussian_centers)
                if not compute_intersection_for_flat_gaussian:
                    warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
                else:
                    closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                    closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][points_closest_gaussians_idx]
                    neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
                neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
                
                if not compute_flat_normals:
                    density_grad = (neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]).sum(dim=-2)
                else:
                    closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                    closest_gaussian_min_scales = self.scaling.min(dim=-1, keepdim=True)[0][points_closest_gaussians_idx]
                    density_grad = (
                        neighbor_opacities[..., None] * 
                        1. / (closest_gaussian_min_scales).pow(2)  * (shift * closest_gaussian_normals).sum(dim=-1, keepdim=True) * closest_gaussian_normals
                        ).sum(dim=-2)
                
                intersection_normals = -torch.nn.functional.normalize(density_grad, dim=-1)
                outputs['normals'] = intersection_normals
                    
            all_outputs[surface_level] = outputs

        return all_outputs
    
    def compute_level_surface_points_from_camera_fast(
        self, 
        nerf_cameras=None,
        cam_idx=0,
        rasterizer=None,
        surface_levels=[0.1, 0.3, 0.5],  # 0.5 or 0.1
        n_surface_points=-1,
        primitive_types=None,  # Should be 'diamond'
        triangle_scale=None,  # Should be 2.
        splat_mesh=True,  # True
        n_points_in_range=21,  # 21
        range_size=3.,  # 3.
        n_points_per_pass=2_000_000,
        density_factor=1.,
        return_pixel_idx=False,
        return_gaussian_idx=False,
        return_normals=False,
        compute_flat_normals=False,
        compute_intersection_for_flat_gaussian=False,  # Should be False
        use_gaussian_depth=False,  # False until now. TODO: Test with True
        just_use_depth_as_level=False, # Should be False
        ):
        # Remember to reset neighbors and update texture features before calling this function
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        
        if primitive_types is not None:
            self.primitive_types = primitive_types
            
        if triangle_scale is not None:
            self.triangle_scale = triangle_scale
            
        if rasterizer is None:
            faces_per_pixel = 10
            max_faces_per_bin = 50_000

            mesh_raster_settings = RasterizationSettings(
                image_size=(self.image_height, self.image_width),
                blur_radius=0.0, 
                faces_per_pixel=faces_per_pixel,
                max_faces_per_bin=max_faces_per_bin
            )
            rasterizer = MeshRasterizer(
                    cameras=nerf_cameras.p3d_cameras[cam_idx], 
                    raster_settings=mesh_raster_settings,
                )
            
        p3d_cameras = nerf_cameras.p3d_cameras[cam_idx]
        
        # Compute splatted depth
        # (either using Gaussian Splatting rasterizer, 
        # or using PyTorch3D's triangle rasterizer for sharper results 
        # and instant access to closest gaussian index for each pixel.)
        if use_gaussian_depth:
            point_depth = p3d_cameras.get_world_to_view_transform().transform_points(self.points)[..., 2:].expand(-1, 3)
            depth = self.render_image_gaussian_rasterizer( 
                    camera_indices=cam_idx,
                    bg_color=torch.Tensor([-1., -1., -1.]).to(self.device),
                    sh_deg=0,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=False,
                    use_same_scale_in_all_directions=False,
                    point_colors=point_depth,
                ).contiguous()[..., 0]
        else:
            if True:
                textures_img = self.get_texture_img(
                    nerf_cameras=nerf_cameras, 
                    cam_idx=cam_idx,
                    sh_levels=self.sh_levels,
                    )

            if splat_mesh:
                mesh = self.splat_mesh(p3d_cameras)
            else:
                mesh = self.mesh
            if True:
                mesh.textures._maps_padded = textures_img[None]

            fragments = rasterizer(mesh, cameras=p3d_cameras)
            depth = fragments.zbuf[0, ..., 0]
        no_depth_mask = depth < 0.
        depth[no_depth_mask] = depth.max() * 1.05
        
        # We backproject the points in world space
        batch_size = 1
        x_tab = torch.Tensor([[i for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        y_tab = torch.Tensor([[j for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
        ndc_x_tab = self.image_width / min(self.image_width,
                                                self.image_height) - (y_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2
        ndc_y_tab = self.image_height / min(self.image_width,
                                                    self.image_height) - (x_tab / (min(self.image_width,
                                                                                    self.image_height) - 1)) * 2

        ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                ndc_y_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                depth.view(batch_size, -1, 1)),
                                dim=-1
                                ).view(batch_size, self.image_height * self.image_width, 3)
        
        fov_cameras = nerf_cameras.p3d_cameras[cam_idx]
        no_proj_mask = no_depth_mask.view(-1)
        ndc_points = ndc_points[0][~no_proj_mask][None]  # Remove pixels with no projection
        if n_surface_points == -1:
            n_surface_points = ndc_points.shape[1]
            ndc_points_idx = torch.arange(n_surface_points)
        else:
            n_surface_points = min(n_surface_points, ndc_points.shape[1])
            ndc_points_idx = torch.randperm(ndc_points.shape[1])[:n_surface_points]
            ndc_points = ndc_points[:, ndc_points_idx]
        all_world_points = fov_cameras.unproject_points(ndc_points, scaled_depth_input=False).view(-1, 3)
        
        # Gather info about gaussians surrounding each 3D point
        if use_gaussian_depth:
            closest_gaussians_idx = self.get_gaussians_closest_to_samples(all_world_points)
            gaussian_idx = closest_gaussians_idx[..., 0]
        else:
            gaussian_idx = fragments.pix_to_face[..., 0].view(-1) // self.n_triangles_per_gaussian
            gaussian_idx = gaussian_idx[~no_proj_mask][ndc_points_idx]
            closest_gaussians_idx = self.knn_idx[gaussian_idx]
        
        # We compute the standard deviation of the gaussian at each point
        gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)
        gaussian_standard_deviations = (self.scaling * quaternion_apply(quaternion_invert(self.quaternions), gaussian_to_camera)).norm(dim=-1)
        points_stds = gaussian_standard_deviations[closest_gaussians_idx[..., 0]]
        
        # We compute ray samples
        points_range = torch.linspace(-range_size, range_size, n_points_in_range).to(self.device).view(1, -1, 1)  # (1, n_points_in_range, 1)
        points_range = points_range * points_stds[..., None, None].expand(-1, n_points_in_range, 1)  # (n_points, n_points_in_range, 1)
        camera_to_samples = torch.nn.functional.normalize(all_world_points - fov_cameras.get_camera_center(), dim=-1)  # (n_points, 3)
        samples = (all_world_points[:, None, :] + points_range * camera_to_samples[:, None, :]).view(-1, 3)  # (n_points * n_points_in_range, 3)
        samples_closest_gaussians_idx = closest_gaussians_idx[:, None, :].expand(-1, n_points_in_range, -1).reshape(-1, self.knn_to_track)
        
        # Compute densities of all samples
        densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
        gaussian_strengths = self.strengths
        gaussian_centers = self.points
        gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True)
        
        for i in range(0, len(samples), n_points_per_pass):
            i_start = i
            i_end = min(len(samples), i + n_points_per_pass)
            
            pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]
            
            closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[pass_closest_gaussians_idx]
            closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]

            shift = (samples[i_start:i_end, None] - closest_gaussian_centers)
            if not compute_intersection_for_flat_gaussian:
                warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
            else:
                closest_gaussian_normals = self.get_normals()[pass_closest_gaussians_idx]
                closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][pass_closest_gaussians_idx]
                neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
            neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
            pass_densities = neighbor_opacities.sum(dim=-1)
            pass_density_mask = pass_densities >= 1.
            pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (pass_densities[pass_density_mask].detach() + 1e-12)
            
            densities[i_start:i_end] = pass_densities
        densities = densities.reshape(-1, n_points_in_range)
        
        # Compute first surface intersection points
        all_outputs = {}
        for surface_level in surface_levels:
            outputs = {}
                        
            under_level = (densities - surface_level < 0)
            above_level = (densities - surface_level > 0)

            _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
            empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

            if not just_use_depth_as_level:
                valid_densities = densities[~empty_pixels]
                valid_range = points_range[~empty_pixels][..., 0]
                valid_first_point_above_level = first_point_above_level[~empty_pixels]

                first_value_above_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level).view(-1)
                value_before_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

                first_t_above_level = valid_range.gather(dim=-1, index=valid_first_point_above_level).view(-1)
                t_before_level = valid_range.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

                intersection_t = (surface_level - value_before_level) / (first_value_above_level - value_before_level) * (first_t_above_level - t_before_level) + t_before_level
                intersection_points = (all_world_points[~empty_pixels] + intersection_t[:, None] * camera_to_samples[~empty_pixels])
            else:
                empty_pixels = torch.zeros_like(empty_pixels, dtype=torch.bool)
                intersection_points = all_world_points[~empty_pixels]
            outputs['intersection_points'] = intersection_points
            
            if return_pixel_idx:
                pixel_idx = torch.arange(self.image_height * self.image_width, dtype=torch.long, device=self.device)
                pixel_idx = pixel_idx[~no_proj_mask][ndc_points_idx][~empty_pixels]
                outputs['pixel_idx'] = pixel_idx
                
            if return_gaussian_idx:
                outputs['gaussian_idx'] = gaussian_idx[~empty_pixels]
            
            if return_normals:                
                points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]

                closest_gaussian_centers = gaussian_centers[points_closest_gaussians_idx]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[points_closest_gaussians_idx]
                closest_gaussian_strengths = gaussian_strengths[points_closest_gaussians_idx]

                shift = (intersection_points[:, None] - closest_gaussian_centers)
                if not compute_intersection_for_flat_gaussian:
                    warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
                else:
                    closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                    closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][points_closest_gaussians_idx]
                    neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
                neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
                
                if not compute_flat_normals:
                    density_grad = (neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]).sum(dim=-2)
                else:
                    closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                    closest_gaussian_min_scales = self.scaling.min(dim=-1, keepdim=True)[0][points_closest_gaussians_idx]
                    density_grad = (
                        neighbor_opacities[..., None] * 
                        1. / (closest_gaussian_min_scales).pow(2)  * (shift * closest_gaussian_normals).sum(dim=-1, keepdim=True) * closest_gaussian_normals
                        ).sum(dim=-2)
                
                intersection_normals = -torch.nn.functional.normalize(density_grad, dim=-1)
                outputs['normals'] = intersection_normals
                    
            all_outputs[surface_level] = outputs

        return all_outputs
    
    def render_image_gaussian_rasterizer(
        self, 
        nerf_cameras:CamerasWrapper=None, 
        camera_indices:int=0,
        verbose=False,
        bg_color = None,
        sh_deg:int=None,
        sh_rotations:torch.Tensor=None,
        compute_color_in_rasterizer=False,
        compute_covariance_in_rasterizer=True,
        return_2d_radii = False,
        quaternions=None,
        use_same_scale_in_all_directions=False,
        return_opacities:bool=False,
        return_colors:bool=False,
        positions:torch.Tensor=None,
        point_colors = None,
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
            nerf_cameras = self.nerfmodel.training_cameras

        p3d_camera = nerf_cameras.p3d_cameras[camera_indices]

        if bg_color is None:
            bg_color = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
            
        if positions is None:
            positions = self.points

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
            self.fov_x, 
            self.fov_y).transpose(0, 1).cuda()
        # TODO: THE TWO FOLLOWING LINES ARE IMPORTANT! IT'S NOT HERE IN 3DGS CODE! Should make a PR when I have time
        proj_transform[..., 2, 0] = - p3d_camera.K[0, 0, 2]
        proj_transform[..., 2, 1] = - p3d_camera.K[0, 1, 2]
        
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_transform.unsqueeze(0))).squeeze(0)
        

        camera_center = p3d_camera.get_camera_center()
        if verbose:
            print("p3d camera_center", camera_center)
            print("ns camera_center", nerf_cameras.camera_to_worlds[camera_indices][..., 3])

        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
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
        if point_colors is None:
            if not compute_color_in_rasterizer:
                if sh_rotations is None:
                    splat_colors = self.get_points_rgb(
                        positions=positions, 
                        camera_centers=camera_center,
                        sh_levels=sh_deg+1,)
                else:
                    splat_colors = self.get_points_rgb(
                        positions=positions, 
                        camera_centers=None,
                        directions=(torch.nn.functional.normalize(positions - camera_center, dim=-1).unsqueeze(1) @ sh_rotations)[..., 0, :],
                        sh_levels=sh_deg+1,)
                shs = None
            else:
                shs = self.sh_coordinates
                splat_colors = None
        else:
            splat_colors = point_colors
            shs = None
            
        splat_opacities = self.strengths.view(-1, 1)
        
        if quaternions is None:
            quaternions = self.quaternions
        
        if not use_same_scale_in_all_directions:
            scales = self.scaling
        else:
            scales = self.scaling.mean(dim=-1, keepdim=True).expand(-1, 3)
            scales = scales.squeeze(0)
        
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
        screenspace_points = torch.zeros(self.n_points, 3, dtype=self._points.dtype, requires_grad=True, device=self.device)
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

    def save_model(self, path, **kwargs):
        checkpoint = {}
        checkpoint['state_dict'] = self.state_dict()
        for k, v in kwargs.items():
            checkpoint[k] = v
        torch.save(checkpoint, path)        


def load_refined_model(refined_sugar_path, nerfmodel:GaussianSplattingWrapper):
    checkpoint = torch.load(refined_sugar_path, map_location=nerfmodel.device)
    n_faces = checkpoint['state_dict']['_surface_mesh_faces'].shape[0]
    n_gaussians = checkpoint['state_dict']['_scales'].shape[0]
    n_gaussians_per_surface_triangle = n_gaussians // n_faces

    print("Loading refined model...")
    print(f'{n_faces} faces detected.')
    print(f'{n_gaussians} gaussians detected.')
    print(f'{n_gaussians_per_surface_triangle} gaussians per surface triangle detected.')

    with torch.no_grad():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(checkpoint['state_dict']['_points'].cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(checkpoint['state_dict']['_surface_mesh_faces'].cpu().numpy())
        # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(checkpoint['state_dict']['_points']).cpu().numpy())
        
    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree+1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )
    refined_sugar.load_state_dict(checkpoint['state_dict'])
    
    return refined_sugar


def load_rc_model(
    nerfmodel, 
    rc_path, 
    initialize=True,
    sh_levels=3,
    learnable_positions=True,
    triangle_scale=1.5,
    retrocompatibility=False,
    use_light_probes=False,
    n_light_probes=1000,
    use_grid_for_light_probes=False,
    ):

    checkpoint = torch.load(rc_path, map_location=nerfmodel.device)
    
    if retrocompatibility:
        if not '_points' in checkpoint['state_dict'].keys():
            checkpoint['state_dict']['_points'] = checkpoint['state_dict']['points']
            checkpoint['state_dict'].pop('points')
            
        if not '_sh_coordinates_dc' in checkpoint['state_dict'].keys():
            checkpoint['state_dict']['_sh_coordinates_dc'] = checkpoint['state_dict']['sh_coordinates'][..., 0:1, :]
            checkpoint['state_dict']['_sh_coordinates_rest'] = checkpoint['state_dict']['sh_coordinates'][..., 1:, :]
            checkpoint['state_dict'].pop('sh_coordinates')
            
        if not '_scales' in checkpoint['state_dict'].keys():
            checkpoint['state_dict']['_scales'] = checkpoint['state_dict']['radiuses'][0, ..., 4:]
            checkpoint['state_dict']['_quaternions'] = checkpoint['state_dict']['radiuses'][0, ..., :4]
            checkpoint['state_dict'].pop('radiuses')
            
        if '_scales' in checkpoint['state_dict'].keys():
            if checkpoint['state_dict']['_scales'].shape[0] == 1:
                checkpoint['state_dict']['_scales'] = checkpoint['state_dict']['_scales'].squeeze(0)
                checkpoint['state_dict']['_quaternions'] = checkpoint['state_dict']['_quaternions'].squeeze(0)
                
    if retrocompatibility:
        checkpoint_state_dict = {}
        checkpoint_state_dict['_points'] = checkpoint['state_dict']['_points']
        checkpoint_state_dict['_sh_coordinates_dc'] = checkpoint['state_dict']['_sh_coordinates_dc']
        checkpoint_state_dict['_sh_coordinates_rest'] = checkpoint['state_dict']['_sh_coordinates_rest']
        checkpoint_state_dict['_scales'] = checkpoint['state_dict']['_scales']
        checkpoint_state_dict['_quaternions'] = checkpoint['state_dict']['_quaternions']
        checkpoint_state_dict['all_densities'] = checkpoint['state_dict']['all_densities']
    
    if not use_light_probes:
        colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
    else:
        colors = 0.5 * torch.ones_like(checkpoint['state_dict']['_points'])
    
    rc = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=colors,
        initialize=initialize,
        sh_levels=sh_levels,
        learnable_positions=learnable_positions,
        triangle_scale=triangle_scale,
        keep_track_of_knn=False,
        knn_to_track=0,
        use_light_probes=use_light_probes,
        n_light_probes=n_light_probes,
        use_grid_for_light_probes=use_grid_for_light_probes,
        )
    
    rc.load_state_dict(checkpoint['state_dict'])
    return rc


def _convert_vertex_colors_to_texture(
    rc:SuGaR, 
    colors:torch.Tensor,
    square_size:int=4,
    ):
    points_to_mesh = rc.points
    
    n_square_per_axis = int(np.sqrt(len(points_to_mesh)) + 1)
    texture_size = square_size * n_square_per_axis
    
    n_features = colors.shape[-1]
    
    point_idx_per_pixel = torch.zeros(texture_size, texture_size, device=rc.device).int()

    with torch.no_grad():
        # Build face UVs
        faces_uv = torch.Tensor(
            [[0, 2, 1], [0, 3, 2]]
            ).to(rc.device)[None] + 4 * torch.arange(len(points_to_mesh), device=rc.device)[:, None, None]
        faces_uv = faces_uv.view(-1, 3).long()

        # Build verts UVs
        verts_coords = torch.cartesian_prod(
            torch.arange(n_square_per_axis, device=rc.device), 
            torch.arange(n_square_per_axis, device=rc.device)
            )[:, None] * square_size
        verts_uv = torch.Tensor(
            [[1., 1.], [1., square_size-1], [square_size-1, square_size-1], [square_size-1, 1.]]
            ).to(rc.device)[None] + verts_coords
        verts_uv = verts_uv.view(-1, 2).long()[:4*len(points_to_mesh)] / texture_size

        # Build texture image
        texture_img = torch.zeros(texture_size, texture_size, n_features, device=rc.device)    
        n_squares_filled = 0
        for i in range(n_square_per_axis):
            for j in range(n_square_per_axis):
                if n_squares_filled >= len(points_to_mesh):
                    break
                start_idx_i = i * square_size
                start_idx_j = j * square_size
                texture_img[..., 
                            start_idx_i:start_idx_i + square_size, 
                            start_idx_j:start_idx_j + square_size, :] = colors[i*n_square_per_axis + j].unsqueeze(0).unsqueeze(0)
                point_idx_per_pixel[...,
                                    start_idx_i:start_idx_i + square_size, 
                                    start_idx_j:start_idx_j + square_size] = i*n_square_per_axis + j
                n_squares_filled += 1
                
        texture_img = texture_img.transpose(0, 1)
        texture_img = texture_img.flip(0)
        
        point_idx_per_pixel = point_idx_per_pixel.transpose(0, 1)
        point_idx_per_pixel = point_idx_per_pixel.flip(0)
    
    return faces_uv, verts_uv, texture_img, point_idx_per_pixel


def extract_texture_image_and_uv_from_gaussians(
    rc:SuGaR,
    square_size:int=10,
    n_sh=-1,
    texture_with_gaussian_renders=True,
    ):
    
    from pytorch3d.renderer import (
    AmbientLights,
    MeshRenderer,
    SoftPhongShader,
    )
    from pytorch3d.renderer.blending import BlendParams
    
    if square_size < 3:
        raise ValueError("square_size must be >= 3")
    
    surface_mesh = rc.surface_mesh
    verts = surface_mesh.verts_list()[0]
    faces = surface_mesh.faces_list()[0]
    faces_verts = verts[faces]
    
    n_triangles = len(faces)
    n_gaussians_per_triangle = rc.n_gaussians_per_surface_triangle
    n_squares = n_triangles // 2 + 1
    n_square_per_axis = int(np.sqrt(n_squares) + 1)
    texture_size = square_size * (n_square_per_axis)
    
    if n_sh==-1:
        n_sh = rc.sh_coordinates.shape[1]
    faces_features = rc.sh_coordinates[:, :n_sh].reshape(n_triangles, n_gaussians_per_triangle, n_sh * 3)
    n_features = faces_features.shape[-1]
    
    if texture_with_gaussian_renders:
        n_features = 3
    
    # Build faces UV.
    # Each face will have 3 corresponding vertices in the UV map
    faces_uv = torch.arange(3 * n_triangles, device=rc.device).view(n_triangles, 3)  # n_triangles, 3
    
    # Build corresponding vertices UV
    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=rc.device), 
        torch.arange(n_square_per_axis, device=rc.device))
    bottom_verts_uv = torch.cat(
        [vertices_uv[n_square_per_axis:-1, None], vertices_uv[:-n_square_per_axis-1, None], vertices_uv[n_square_per_axis+1:, None]],
        dim=1)
    top_verts_uv = torch.cat(
        [vertices_uv[1:-n_square_per_axis, None], vertices_uv[:-n_square_per_axis-1, None], vertices_uv[n_square_per_axis+1:, None]],
        dim=1)
    
    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=rc.device), 
        torch.arange(n_square_per_axis, device=rc.device))[:, None]
    u_shift = torch.tensor([[1, 0]], dtype=torch.int32, device=rc.device)[:, None]
    v_shift = torch.tensor([[0, 1]], dtype=torch.int32, device=rc.device)[:, None]
    bottom_verts_uv = torch.cat(
        [vertices_uv + u_shift, vertices_uv, vertices_uv + u_shift + v_shift],
        dim=1)
    top_verts_uv = torch.cat(
        [vertices_uv + v_shift, vertices_uv, vertices_uv + u_shift + v_shift],
        dim=1)
    
    verts_uv = torch.cat([bottom_verts_uv, top_verts_uv], dim=1)
    verts_uv = verts_uv * square_size
    verts_uv[:, 0] = verts_uv[:, 0] + torch.tensor([[-2, 1]], device=rc.device)
    verts_uv[:, 1] = verts_uv[:, 1] + torch.tensor([[2, 1]], device=rc.device)
    verts_uv[:, 2] = verts_uv[:, 2] + torch.tensor([[-2, -3]], device=rc.device)
    verts_uv[:, 3] = verts_uv[:, 3] + torch.tensor([[1, -1]], device=rc.device)
    verts_uv[:, 4] = verts_uv[:, 4] + torch.tensor([[1, 3]], device=rc.device)
    verts_uv[:, 5] = verts_uv[:, 5] + torch.tensor([[-3, -1]], device=rc.device)
    verts_uv = verts_uv.reshape(-1, 2) / texture_size
    
    # ---Build texture image
    # Start by computing pixel indices for each triangle
    texture_img = torch.zeros(texture_size, texture_size, n_features, device=rc.device)    
    pixel_idx_inside_bottom_triangle = torch.zeros(0, 2, dtype=torch.int32, device=rc.device)
    pixel_idx_inside_top_triangle = torch.zeros(0, 2, dtype=torch.int32, device=rc.device)
    for tri_i in range(0, square_size-1):
        for tri_j in range(0, tri_i+1):
            pixel_idx_inside_bottom_triangle = torch.cat(
                [pixel_idx_inside_bottom_triangle, torch.tensor([[tri_i, tri_j]], dtype=torch.int32, device=rc.device)], dim=0)
    for tri_i in range(0, square_size):
        for tri_j in range(tri_i+1, square_size):
            pixel_idx_inside_top_triangle = torch.cat(
                [pixel_idx_inside_top_triangle, torch.tensor([[tri_i, tri_j]], dtype=torch.int32, device=rc.device)], dim=0)
    
    bottom_triangle_pixel_idx = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=rc.device), 
        torch.arange(n_square_per_axis, device=rc.device))[:, None] * square_size + pixel_idx_inside_bottom_triangle[None]
    top_triangle_pixel_idx = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=rc.device), 
        torch.arange(n_square_per_axis, device=rc.device))[:, None] * square_size + pixel_idx_inside_top_triangle[None]
    triangle_pixel_idx = torch.cat(
        [bottom_triangle_pixel_idx[:, None], 
        top_triangle_pixel_idx[:, None]],
        dim=1).view(-1, bottom_triangle_pixel_idx.shape[-2], 2)[:n_triangles]
    
    # Then we compute the barycentric coordinates of each pixel inside its corresponding triangle
    bottom_triangle_pixel_bary_coords = pixel_idx_inside_bottom_triangle.clone().float()
    bottom_triangle_pixel_bary_coords[..., 0] = -(bottom_triangle_pixel_bary_coords[..., 0] - (square_size - 2))
    bottom_triangle_pixel_bary_coords[..., 1] = (bottom_triangle_pixel_bary_coords[..., 1] - 1)
    bottom_triangle_pixel_bary_coords = (bottom_triangle_pixel_bary_coords + 0.) / (square_size - 3)
    bottom_triangle_pixel_bary_coords = torch.cat(
        [1. - bottom_triangle_pixel_bary_coords.sum(dim=-1, keepdim=True), bottom_triangle_pixel_bary_coords],
        dim=-1)
    top_triangle_pixel_bary_coords = pixel_idx_inside_top_triangle.clone().float()
    top_triangle_pixel_bary_coords[..., 0] = (top_triangle_pixel_bary_coords[..., 0] - 1)
    top_triangle_pixel_bary_coords[..., 1] = -(top_triangle_pixel_bary_coords[..., 1] - (square_size - 1))
    top_triangle_pixel_bary_coords = (top_triangle_pixel_bary_coords + 0.) / (square_size - 3)
    top_triangle_pixel_bary_coords = torch.cat(
        [1. - top_triangle_pixel_bary_coords.sum(dim=-1, keepdim=True), top_triangle_pixel_bary_coords],
        dim=-1)
    triangle_pixel_bary_coords = torch.cat(
        [bottom_triangle_pixel_bary_coords[None],
        top_triangle_pixel_bary_coords[None]],
        dim=0)  # 2, n_pixels_per_triangle, 3
    
    all_triangle_bary_coords = triangle_pixel_bary_coords[None].expand(n_squares, -1, -1, -1).reshape(-1, triangle_pixel_bary_coords.shape[-2], 3)
    all_triangle_bary_coords = all_triangle_bary_coords[:len(faces_verts)]
    
    pixels_space_positions = (all_triangle_bary_coords[..., None] * faces_verts[:, None]).sum(dim=-2)[:, :, None]
    
    gaussian_centers = rc.points.reshape(-1, 1, rc.n_gaussians_per_surface_triangle, 3)
    gaussian_inv_scaled_rotation = rc.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True).reshape(-1, 1, rc.n_gaussians_per_surface_triangle, 3, 3)
    
    # Compute the density field as a sum of local gaussian opacities
    shift = (pixels_space_positions - gaussian_centers)
    warped_shift = gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
    neighbor_opacities = torch.exp(-1. / 2 * neighbor_opacities) # / rc.n_gaussians_per_surface_triangle
    
    pixel_features = faces_features[:, None].expand(-1, neighbor_opacities.shape[1], -1, -1).gather(
        dim=-2,
        index=neighbor_opacities[..., None].argmax(dim=-2, keepdim=True).expand(-1, -1, -1, 3)
        )[:, :, 0, :]
        
    # pixel_alpha = neighbor_opacities.sum(dim=-1, keepdim=True)
    texture_img[(triangle_pixel_idx[..., 0], triangle_pixel_idx[..., 1])] = pixel_features

    texture_img = texture_img.transpose(0, 1)
    texture_img = SH2RGB(texture_img.flip(0))
    
    faces_per_pixel = 1
    max_faces_per_bin = 50_000
    mesh_raster_settings = RasterizationSettings(
        image_size=(rc.image_height, rc.image_width),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
        # max_faces_per_bin=max_faces_per_bin
    )
    lights = AmbientLights(device=rc.device)
    rasterizer = MeshRasterizer(
            cameras=rc.nerfmodel.training_cameras.p3d_cameras[0], 
            raster_settings=mesh_raster_settings,
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=rc.device, 
            cameras=rc.nerfmodel.training_cameras.p3d_cameras[0],
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        )
    )
    texture_idx = torch.cartesian_prod(
        torch.arange(texture_size, device=rc.device), 
        torch.arange(texture_size, device=rc.device)
        ).reshape(texture_size, texture_size, 2
                    )
    texture_idx = torch.cat([texture_idx, torch.zeros_like(texture_idx[..., 0:1])], dim=-1)
    texture_counter = torch.zeros(texture_size, texture_size, 1, device=rc.device)
    idx_textures_uv = TexturesUV(
        maps=texture_idx[None].float(), #texture_img[None]),
        verts_uvs=verts_uv[None],
        faces_uvs=faces_uv[None],
        sampling_mode='nearest',
        )
    idx_mesh = Meshes(
        verts=[rc.surface_mesh.verts_list()[0]],   
        faces=[rc.surface_mesh.faces_list()[0]],
        textures=idx_textures_uv,
        )
    
    for cam_idx in range(len(rc.nerfmodel.training_cameras)):
        p3d_cameras = rc.nerfmodel.training_cameras.p3d_cameras[cam_idx]
        
        # Render rgb img
        rgb_img = rc.render_image_gaussian_rasterizer(
            camera_indices=cam_idx,
            sh_deg=0,  #rc.sh_levels-1,
            compute_color_in_rasterizer=True,  #compute_color_in_rasterizer,
        ).clamp(min=0, max=1)
        
        fragments = renderer.rasterizer(idx_mesh, cameras=p3d_cameras)
        idx_img = renderer.shader(fragments, idx_mesh, cameras=p3d_cameras)[0, ..., :2]
        # print("Idx img:", idx_img.shape, idx_img.min(), idx_img.max())
        update_mask = fragments.zbuf[0, ..., 0] > 0
        idx_to_update = idx_img[update_mask].round().long() 

        use_average = True
        if not use_average:
            texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = rgb_img[update_mask]
        else:
            no_initialize_mask = texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])][..., 0] != 0
            texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = no_initialize_mask[..., None] * texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])]

            texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] + rgb_img[update_mask]
            texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])] = texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])] + 1

    if use_average:
        texture_img = texture_img / texture_counter.clamp(min=1)        
    
    return verts_uv, faces_uv, texture_img


def convert_refined_sugar_into_gaussians(refined_sugar):
    new_gaussians = GaussianModel(refined_sugar.sh_levels - 1)
    
    with torch.no_grad():
        xyz = refined_sugar.points.cpu().numpy()
        opacities = refined_sugar.all_densities.cpu().numpy()
        features_dc = refined_sugar._sh_coordinates_dc.permute(0, 2, 1).cpu().numpy()
        features_extra = refined_sugar._sh_coordinates_rest.permute(0, 2, 1).cpu().numpy()
        
        scales = scale_inverse_activation(refined_sugar.scaling).cpu().numpy()
        rots = refined_sugar.quaternions.cpu().numpy()

    new_gaussians._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    new_gaussians._features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    new_gaussians._features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    new_gaussians._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    new_gaussians._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    new_gaussians._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    new_gaussians.active_sh_degree = new_gaussians.max_sh_degree
    
    return new_gaussians
