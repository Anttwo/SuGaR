from typing import Union
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings as P3DRasterizationSettings
from pytorch3d.renderer import MeshRasterizer as P3DMeshRasterizer
try:
    # Ensure the NVDIFRAST_BACKEND is set to 'PACKED' if using the CUDA rasterizer
    if os.environ.get('NVDIFRAST_CUDA_RASTERIZER', '0') in ('1', 'True', 'true'):
        os.environ.setdefault('NVDIFRAST_BACKEND', 'PACKED')

    from .nvdiffrast import nvdiff_rasterization, dr
    nvdiffrast_available = True
except ImportError:
    print("Nvdiffrast not found. Using PyTorch3D rasterizer instead.\n"
        "With Nvdiffrast, the rendering is much faster; For instance, computing a" 
        "texture for a mesh using a collection of images can be done in just a few seconds.\n"
        "Without Nvdiffrast, the rendering is much slower and computing the texture can take a few minutes."
    )
    dr = None
    nvdiffrast_available = False
from sugar_scene.cameras import CamerasWrapper, P3DCameras, GSCamera


class RasterizationSettings():
    def __init__(
        self, 
        image_size=(1080, 1920),
        blur_radius=0.0,
        faces_per_pixel=1,
        ):
        self.image_size = image_size        
        self._p3d_raster_settings = P3DRasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        )


class Fragments():
    def __init__(self, bary_coords, zbuf, pix_to_face):
        self.bary_coords = bary_coords  # Shape (1, height, width, 1, 3)
        self.zbuf = zbuf  # Shape (1, height, width, 1)
        self.pix_to_face = pix_to_face  # Shape (1, height, width, 1)


class MeshRasterizer(torch.nn.Module):
    def __init__(
        self, 
        cameras:Union[P3DCameras, GSCamera, CamerasWrapper]=None,
        raster_settings:RasterizationSettings=None,
        use_nvdiffrast:bool=True,
    ):
        super().__init__()
        
        if not nvdiffrast_available:
            use_nvdiffrast = False
        self.use_nvdiffrast = use_nvdiffrast
        
        # Get height and width if provided in cameras
        if isinstance(cameras, CamerasWrapper):
            self.height = cameras.gs_cameras[0].image_height
            self.width = cameras.gs_cameras[0].image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = cameras
            
        elif isinstance(cameras, GSCamera):
            self.height = cameras.image_height
            self.width = cameras.image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = CamerasWrapper(gs_cameras=[cameras])
            
        elif isinstance(cameras, list) and isinstance(cameras[0], GSCamera):
            self.height = cameras[0].image_height
            self.width = cameras[0].image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = CamerasWrapper(gs_cameras=cameras)
        
        elif isinstance(cameras, P3DCameras):
            if raster_settings is None:
                raster_settings = RasterizationSettings()
            self.raster_settings = raster_settings
            self.height, self.width = raster_settings.image_size
            self.cameras = CamerasWrapper.from_p3d_cameras(
                p3d_cameras=cameras, 
                height=self.height, 
                width=self.width,
            )            
            
        elif cameras is None:
            if raster_settings is None:
                raster_settings = RasterizationSettings()
            self.raster_settings = raster_settings
            self.height, self.width = raster_settings.image_size
            self.cameras = None
        
        else:
            raise ValueError("cameras must be either CamerasWrapper, P3DCameras, GSCamera or list of GSCamera")

        if self.use_nvdiffrast:
            use_cuda_rasterizer = os.environ.get('NVDIFRAST_CUDA_RASTERIZER', '0') in ('1', 'True', 'true')

            if use_cuda_rasterizer:
                self.gl_context = dr.RasterizeCudaContext()
            else:
                self.gl_context = dr.RasterizeGLContext()
        else:
            self._p3d_mesh_rasterizer = P3DMeshRasterizer(
                cameras=self.cameras.p3d_cameras,
                raster_settings=self.raster_settings._p3d_raster_settings,
            )
            
    def forward(
        self, 
        mesh:Meshes, 
        cameras:Union[CamerasWrapper, P3DCameras, GSCamera]=None,
        cam_idx=0,
        return_only_pix_to_face=False,
    ):
        if cameras is None:
            if self.cameras is None:
                raise ValueError("cameras must be provided either in the constructor or in the forward method")
            cameras = self.cameras
        
        if self.use_nvdiffrast:
            if isinstance(cameras, CamerasWrapper):
                render_camera = cameras.gs_cameras[cam_idx]
            elif isinstance(cameras, GSCamera):
                render_camera = cameras
            elif isinstance(cameras, list) and isinstance(cameras[0], GSCamera):
                render_camera = cameras[cam_idx]
            elif isinstance(cameras, P3DCameras):
                render_camera = CamerasWrapper.from_p3d_cameras(
                    p3d_cameras=cameras, 
                    height=self.height, 
                    width=self.width,
                ).gs_cameras[cam_idx]
            else:
                raise ValueError("cameras must be either CamerasWrapper, P3DCameras, GSCamera or list of GSCamera")

            height, width = render_camera.image_height, render_camera.image_width
            if self.use_cuda_rasterizer:
                # Prepare vertex positions in homogeneous coordinates
                verts = mesh.verts_padded()  # Shape: [batch_size, num_verts, 3]
                batch_size, num_verts, _ = verts.shape
                w = torch.ones((batch_size, num_verts, 1), device=verts.device)
                pos = torch.cat([verts, w], dim=-1)  # Shape: [batch_size, num_verts, 4]
                faces = mesh.faces_padded()
                bary_coords, zbuf, pix_to_face = nvdiff_rasterization(
                    camera=render_camera,
                    image_height=height,
                    image_width=width,
                    pos=pos,
                    faces=faces,
                    return_indices_only=False,
                    glctx=self.gl_context,
                )
            else:
                bary_coords, zbuf, pix_to_face = nvdiff_rasterization(
                    camera=render_camera,
                    image_height=height,
                    image_width=width,
                    mesh=mesh,
                    return_indices_only=False,
                    glctx=self.gl_context,
                )
            pix_to_face = pix_to_face - 1
            if return_only_pix_to_face:
                return pix_to_face.view(1, height, width, 1)
            bary_coords = torch.cat([bary_coords, 1. - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
            
            # TODO: Zbuf is still in NDC space, should convert to camera space
            return Fragments(
                bary_coords.view(1, height, width, 1, 3),
                zbuf.view(1, height, width, 1),
                pix_to_face.view(1, height, width, 1),
            )
        
        else:
            if isinstance(cameras, CamerasWrapper):
                p3d_cameras = cameras.p3d_cameras
            elif isinstance(cameras, GSCamera):
                p3d_cameras = CamerasWrapper(gs_cameras=[cameras]).p3d_cameras
            elif isinstance(cameras, list) and isinstance(cameras[0], GSCamera):
                p3d_cameras = CamerasWrapper(gs_cameras=cameras).p3d_cameras
            elif isinstance(cameras, P3DCameras):
                p3d_cameras = cameras
            else:
                raise ValueError("cameras must be either CamerasWrapper or P3DCameras")
            fragments = self._p3d_mesh_rasterizer(mesh, cameras=p3d_cameras[cam_idx])
            if return_only_pix_to_face:
                return fragments.pix_to_face
            return fragments