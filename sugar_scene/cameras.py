import os
import json

import numpy as np
import torch
from PIL import Image

from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

from sugar_utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
from sugar_utils.general_utils import PILtoTorch


def load_gs_cameras(source_path, gs_output_path, image_resolution=1, 
                    load_gt_images=True, max_img_size=1920):
    """Loads Gaussian Splatting camera parameters from a COLMAP reconstruction.

    Args:
        source_path (str): Path to the source data.
        gs_output_path (str): Path to the Gaussian Splatting output.
        image_resolution (int, optional): Factor by which to downscale the images. Defaults to 1.
        load_gt_images (bool, optional): If True, loads the ground truth images. Defaults to True.
        max_img_size (int, optional): Maximum size of the images. Defaults to 1920.

    Returns:
        List of GSCameras: List of Gaussian Splatting cameras.
    """
    image_dir = os.path.join(source_path, 'images')
    
    with open(gs_output_path + 'cameras.json') as f:
        unsorted_camera_transforms = json.load(f)
    camera_transforms = sorted(unsorted_camera_transforms.copy(), key = lambda x : x['img_name'])

    cam_list = []
    extension = '.' + os.listdir(image_dir)[0].split('.')[-1]
    if extension not in ['.jpg', '.png', '.JPG', '.PNG']:
        print(f"Warning: image extension {extension} not supported.")
    else:
        print(f"Found image extension {extension}")
    
    for cam_idx in range(len(camera_transforms)):
        camera_transform = camera_transforms[cam_idx]
        
        # Extrinsics
        rot = np.array(camera_transform['rotation'])
        pos = np.array(camera_transform['position'])
        
        W2C = np.zeros((4,4))
        W2C[:3, :3] = rot
        W2C[:3, 3] = pos
        W2C[3,3] = 1
        
        Rt = np.linalg.inv(W2C)
        T = Rt[:3, 3]
        R = Rt[:3, :3].transpose()
        
        # Intrinsics
        width = camera_transform['width']
        height = camera_transform['height']
        fy = camera_transform['fy']
        fx = camera_transform['fx']
        fov_y = focal2fov(fy, height)
        fov_x = focal2fov(fx, width)
        
        # GT data
        id = camera_transform['id']
        name = camera_transform['img_name']
        image_path = os.path.join(image_dir,  name + extension)
        
        if load_gt_images:
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            downscale_factor = 1
            if image_resolution in [1, 2, 4, 8]:
                downscale_factor = image_resolution
                # resolution = round(orig_w/(image_resolution)), round(orig_h/(image_resolution))
            if max(orig_h, orig_w) > max_img_size:
                additional_downscale_factor = max(orig_h, orig_w) / max_img_size
                downscale_factor = additional_downscale_factor * downscale_factor
            resolution = round(orig_w/(downscale_factor)), round(orig_h/(downscale_factor))
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            
            image_height, image_width = None, None
        else:
            gt_image = None
            if image_resolution in [1, 2, 4, 8]:
                downscale_factor = image_resolution
                # resolution = round(orig_w/(image_resolution)), round(orig_h/(image_resolution))
            if max(height, width) > max_img_size:
                additional_downscale_factor = max(height, width) / max_img_size
                downscale_factor = additional_downscale_factor * downscale_factor
            image_height, image_width = round(height/downscale_factor), round(width/downscale_factor)
        
        gs_camera = GSCamera(
            colmap_id=id, image=gt_image, gt_alpha_mask=None,
            R=R, T=T, FoVx=fov_x, FoVy=fov_y,
            image_name=name, uid=id,
            image_height=image_height, image_width=image_width,)
        
        cam_list.append(gs_camera)

    return cam_list


class GSCamera(torch.nn.Module):
    """Class to store Gaussian Splatting camera parameters.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_height=None, image_width=None,
                 ):
        """
        Args:
            colmap_id (int): ID of the camera in the COLMAP reconstruction.
            R (np.array): Rotation matrix.
            T (np.array): Translation vector.
            FoVx (float): Field of view in the x direction.
            FoVy (float): Field of view in the y direction.
            image (np.array): GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(GSCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is None:
            if image_height is None or image_width is None:
                raise ValueError("Either image or image_height and image_width must be specified")
            else:
                self.image_height = image_height
                self.image_width = image_width
        else:        
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self

    
def create_p3d_cameras(R=None, T=None, K=None, znear=0.0001):
    """Creates pytorch3d-compatible camera object from R, T, K matrices.

    Args:
        R (torch.Tensor, optional): Rotation matrix. Defaults to Identity.
        T (torch.Tensor, optional): Translation vector. Defaults to Zero.
        K (torch.Tensor, optional): Camera intrinsics. Defaults to None.
        znear (float, optional): Near clipping plane. Defaults to 0.0001.

    Returns:
        pytorch3d.renderer.cameras.FoVPerspectiveCameras: pytorch3d-compatible camera object.
    """
    if R is None:
        R = torch.eye(3)[None]
    if T is None:
        T = torch.zeros(3)[None]
        
    if K is not None:
        p3d_cameras = P3DCameras(R=R, T=T, K=K, znear=0.0001)
    else:
        p3d_cameras = P3DCameras(R=R, T=T, znear=0.0001)
        p3d_cameras.K = p3d_cameras.get_projection_transform().get_matrix().transpose(-1, -2)
        
    return p3d_cameras


def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = (
        -(
            torch.Tensor(
                (cx[0], cy[0]),
            )[
                None
            ].to(device)
            - c0
        )
        / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras


def convert_camera_from_pytorch3d_to_gs(
    p3d_cameras: P3DCameras,
    height: float,
    width: float,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]

    # Inverse intrinsics
    image_size = torch.Tensor(
        [width, height],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    K_inv = p3d_cameras.K[0] * scale
    fx_inv, fy_inv = K_inv[0, 0], K_inv[1, 1]
    cx_inv, cy_inv = c0[0, 0] - K_inv[0, 2], c0[0, 1] - K_inv[1, 2]
    
    gs_cameras = []
    
    for cam_idx in range(N):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        image_height=height
        image_width=width
        
        fx = fx_inv.item()
        fy = fy_inv.item()
        fovx = focal2fov(fx, image_width)
        fovy = focal2fov(fy, image_height)

        FovY = fovy 
        FovX = fovx
        
        name = 'image_' + str(cam_idx)
        
        camera = GSCamera(
            colmap_id=cam_idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=FovX, FoVy=FovY,
            image_name=name, uid=cam_idx,
            image_height=image_height, 
            image_width=image_width,
            )
        gs_cameras.append(camera)

    return gs_cameras


class CamerasWrapper:
    """Class to wrap Gaussian Splatting camera parameters 
    and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(
        self,
        gs_cameras,
        p3d_cameras=None,
        p3d_cameras_computed=False,
    ) -> None:
        """
        Args:
            camera_to_worlds (_type_): _description_
            fx (_type_): _description_
            fy (_type_): _description_
            cx (_type_): _description_
            cy (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
            distortion_params (_type_): _description_
            camera_type (_type_): _description_
        """

        self.gs_cameras = gs_cameras
        
        self._p3d_cameras = p3d_cameras
        self._p3d_cameras_computed = p3d_cameras_computed
        
        device = gs_cameras[0].device        
        N = len(gs_cameras)
        R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
        T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.cx = self.width / 2.  # torch.zeros_like(fx).to(device)
        self.cy = self.height / 2.  # torch.zeros_like(fy).to(device)
        
        w2c = torch.zeros(N, 4, 4).to(device)
        w2c[:, :3, :3] = R.transpose(-1, -2)
        w2c[:, :3, 3] = T
        w2c[:, 3, 3] = 1
        
        c2w = w2c.inverse()
        c2w[:, :3, 1:3] *= -1
        c2w = c2w[:, :3, :]
        self.camera_to_worlds = c2w

    @classmethod
    def from_p3d_cameras(
        cls,
        p3d_cameras,
        width: float,
        height: float,
    ) -> None:
        """Initializes CamerasWrapper from pytorch3d-compatible camera object.

        Args:
            p3d_cameras (_type_): _description_
            width (float): _description_
            height (float): _description_

        Returns:
            _type_: _description_
        """
        cls._p3d_cameras = p3d_cameras
        cls._p3d_cameras_computed = True

        gs_cameras = convert_camera_from_pytorch3d_to_gs(
            p3d_cameras,
            height=height,
            width=width,
        )

        return cls(
            gs_cameras=gs_cameras,
            p3d_cameras=p3d_cameras,
            p3d_cameras_computed=True,
        )

    @property
    def device(self):
        return self.camera_to_worlds.device

    @property
    def p3d_cameras(self):
        if not self._p3d_cameras_computed:
            self._p3d_cameras = convert_camera_from_gs_to_pytorch3d(
                self.gs_cameras,
            )
            self._p3d_cameras_computed = True

        return self._p3d_cameras

    def __len__(self):
        return len(self.gs_cameras)

    def to(self, device):
        self.camera_to_worlds = self.camera_to_worlds.to(device)
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.cx = self.cx.to(device)
        self.cy = self.cy.to(device)
        self.width = self.width.to(device)
        self.height = self.height.to(device)
        
        for gs_camera in self.gs_cameras:
            gs_camera.to(device)

        if self._p3d_cameras_computed:
            self._p3d_cameras = self._p3d_cameras.to(device)

        return self
        
    def get_spatial_extent(self):
        """Returns the spatial extent of the cameras, computed as 
        the extent of the bounding box containing all camera centers.

        Returns:
            (float): Spatial extent of the cameras.
        """
        camera_centers = self.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        return radius
    