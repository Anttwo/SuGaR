import torch
import numpy as np
from sugar_utils.general_utils import get_expon_lr_func
from .sugar_model import SuGaR


class OptimizationParams():
    def __init__(self, 
                 iterations:int=30_000,
                 position_lr_init:float=0.00016,
                 position_lr_final:float=0.0000016,
                 position_lr_delay_mult:float=0.01,
                 position_lr_max_steps:int=30_000,
                 feature_lr:float=0.0025,
                 opacity_lr:float=0.05,
                 scaling_lr:float=0.005,
                 rotation_lr:float=0.001,
                 ):
        
        # Basic Gaussian Splatting
        self.iterations = iterations
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr

    def __str__(self):
        return f"""OptimizationParams(
            iterations={self.iterations},
            position_lr_init={self.position_lr_init},
            position_lr_final={self.position_lr_final},
            position_lr_delay_mult={self.position_lr_delay_mult},
            position_lr_max_steps={self.position_lr_max_steps},
            feature_lr={self.feature_lr},
            opacity_lr={self.opacity_lr},
            scaling_lr={self.scaling_lr},
            rotation_lr={self.rotation_lr},
            )"""


class SuGaROptimizer():
    """Wrapper of the Adam optimizer used for SuGaR optimization.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(
        self,
        model:SuGaR,
        opt:OptimizationParams=None,
        spatial_lr_scale:float=None,
        ) -> None:

        self.current_iteration = 0
        self.num_iterations = opt.iterations
        
        if opt is None:
            opt = OptimizationParams()
        
        if spatial_lr_scale is None:
            spatial_lr_scale = model.get_cameras_spatial_extent()
        self.spatial_lr_scale = spatial_lr_scale
        
        if (not model.binded_to_surface_mesh and model.learn_positions) or (model.binded_to_surface_mesh and model.learn_surface_mesh_positions):
            l = [{'params': [model._points], 'lr': opt.position_lr_init * spatial_lr_scale, "name": "points"}]
        else:
            l = []

        if not model.freeze_gaussians:
            l = l + [
                {'params': [model._sh_coordinates_dc], 'lr': opt.feature_lr, "name": "sh_coordinates_dc"},
                {'params': [model._sh_coordinates_rest], 'lr': opt.feature_lr / 20.0, "name": "sh_coordinates_rest"}
                ]
            
        if model.learn_opacities:
            l = l + [{'params': [model.all_densities], 'lr': opt.opacity_lr, "name": "all_densities"}]
        if (model.binded_to_surface_mesh and model.learn_surface_mesh_scales) or (not model.binded_to_surface_mesh and model.learn_scales):
            l = l + [{'params': [model._scales], 'lr': opt.scaling_lr, "name": "scales"}]
        if (model.binded_to_surface_mesh and model.learn_surface_mesh_scales) or (not model.binded_to_surface_mesh and model.learn_quaternions):
            l = l + [{'params': [model._quaternions], 'lr': opt.rotation_lr, "name": "quaternions"}]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.position_sheduler_func = get_expon_lr_func(
            lr_init=opt.position_lr_init * spatial_lr_scale, 
            lr_final=opt.position_lr_final * spatial_lr_scale, 
            lr_delay_mult=opt.position_lr_delay_mult, 
            max_steps=opt.position_lr_max_steps
            )
        
    def step(self):
        self.optimizer.step()
        self.current_iteration += 1
        
    def zero_grad(self, set_to_none:bool=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
    def update_learning_rate(self, iteration:int=None):
        if iteration is None:
            iteration = self.current_iteration
        lr = 0.
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "points":
                lr = self.position_sheduler_func(iteration)
                param_group['lr'] = lr                
        return lr
            
    def add_param_group(self, new_param_group):
        self.optimizer.add_param_group(new_param_group)

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
