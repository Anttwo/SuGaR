import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from .sugar_model import SuGaR
from .sugar_optimizer import SuGaROptimizer
from sugar_utils.general_utils import inverse_sigmoid


class SuGaRDensifier():
    """Wrapper of the densification functions used for Gaussian Splatting and SuGaR optimization.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(
        self,
        sugar_model:SuGaR,
        sugar_optimizer:SuGaROptimizer,
        max_grad=0.0002,
        min_opacity:float=0.005,
        max_screen_size:int=20,
        scene_extent:float=None,
        percent_dense:float=0.01,
        ) -> None:
        pass
        
        self.model = sugar_model
        self.optimizer = sugar_optimizer.optimizer
                
        self.points_gradient_accum = torch.zeros((self.model.points.shape[0], 1), device=self.model.device)
        self.denom = torch.zeros((self.model.points.shape[0], 1), device=self.model.device)
        self.max_radii2D = torch.zeros((self.model.points.shape[0]), device=self.model.device)
        
        self.max_grad = max_grad
        self.min_opacity = min_opacity
        self.max_screen_size = max_screen_size
        if scene_extent is None:
            self.scene_extent = sugar_model.get_cameras_spatial_extent()
        else:
            self.scene_extent = scene_extent
        self.percent_dense = percent_dense
        
        self.params_to_densify = []        
        if not self.model.freeze_gaussians:
            self.params_to_densify.extend(["points", "all_densities", "scales", "quaternions"])
            self.params_to_densify.extend(["sh_coordinates_dc", "sh_coordinates_rest"])
        
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in self.params_to_densify:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
            
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        
        if "points" in self.params_to_densify:
            self.model._points = optimizable_tensors["points"]
        if "all_densities" in self.params_to_densify:
            self.model.all_densities = optimizable_tensors["all_densities"]
        if "scales" in self.params_to_densify:
            self.model._scales = optimizable_tensors["scales"]
        if "quaternions" in self.params_to_densify:
            self.model._quaternions = optimizable_tensors["quaternions"]
        if "sh_coordinates_dc" in self.params_to_densify:
            self.model._sh_coordinates_dc = optimizable_tensors["sh_coordinates_dc"]
        if "sh_coordinates_rest" in self.params_to_densify:
            self.model._sh_coordinates_rest = optimizable_tensors["sh_coordinates_rest"]
            
        self.points_gradient_accum = self.points_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in self.params_to_densify:
                assert len(group["params"]) == 1
                            
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                dim_to_cat = 0
                        
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=dim_to_cat)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=dim_to_cat)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=dim_to_cat).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=dim_to_cat).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def densification_postfix(self, new_points, 
                              new_densities, new_scales, new_quaternions,
                              new_sh_coordinates_dc=None, new_sh_coordinates_rest=None, 
                              ):
        tensors_dict = {
            "points": new_points,
            "all_densities": new_densities,
            "scales": new_scales,
            "quaternions": new_quaternions
            }
        tensors_dict["sh_coordinates_dc"] = new_sh_coordinates_dc
        tensors_dict["sh_coordinates_rest"] = new_sh_coordinates_rest

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)
        
        self.model._points = optimizable_tensors["points"]
        self.model.all_densities = optimizable_tensors["all_densities"]
        self.model._scales = optimizable_tensors["scales"]
        self.model._quaternions = optimizable_tensors["quaternions"]
        self.model._sh_coordinates_dc = optimizable_tensors["sh_coordinates_dc"]
        self.model._sh_coordinates_rest = optimizable_tensors["sh_coordinates_rest"]

        self.points_gradient_accum = torch.zeros((self.model.points.shape[0], 1), device=self.model.device)
        self.denom = torch.zeros((self.model.points.shape[0], 1), device=self.model.device)
        self.max_radii2D = torch.zeros((self.model.points.shape[0]), device=self.model.device)
    
    def update_densification_stats(self, viewspace_point_tensor, radii, visibility_filter):
        # Updates maximum observed 2D radii of all gaussians
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
        
        # Accumulate gradient magnitudes of all points
        self.points_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)
        
        # Counts number of updates for each point
        self.denom[visibility_filter] += 1

    def densify_and_clone(self, grads, max_grad, extent):
        max_grad = self.max_grad if max_grad is None else max_grad
        extent = self.scene_extent if extent is None else extent
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.scaling, dim=1).values <= self.percent_dense * extent)
        
        new_points = self.model._points[selected_pts_mask]
        new_densities = self.model.all_densities[selected_pts_mask]
        new_scales = self.model._scales[selected_pts_mask]
        new_quaternions = self.model._quaternions[selected_pts_mask]
        new_sh_coordinates_dc = self.model._sh_coordinates_dc[selected_pts_mask]
        new_sh_coordinates_rest = self.model._sh_coordinates_rest[selected_pts_mask]
        
        self.densification_postfix(
            new_points=new_points,
            new_densities=new_densities, 
            new_scales=new_scales, 
            new_quaternions=new_quaternions,
            new_sh_coordinates_dc=new_sh_coordinates_dc, 
            new_sh_coordinates_rest=new_sh_coordinates_rest,
        )
    
    def densify_and_split(self, grads, max_grad, extent, N=2):
        max_grad = self.max_grad if max_grad is None else max_grad
        extent = self.scene_extent if extent is None else extent
        
        n_init_points = self.model._points.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.scaling, dim=1).values > self.percent_dense*extent)

        stds = self.model.scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self.model.quaternions[selected_pts_mask]).repeat(N, 1, 1)
        
        new_points = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.points[selected_pts_mask].repeat(N, 1)
        new_scales = self.model.scale_inverse_activation(self.model.scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_quaternions = self.model._quaternions[selected_pts_mask].repeat(N,1)
        new_densities = self.model.all_densities[selected_pts_mask].repeat(N,1)
        new_sh_coordinates_dc = self.model._sh_coordinates_dc[selected_pts_mask].repeat(N,1,1)
        new_sh_coordinates_rest = self.model._sh_coordinates_rest[selected_pts_mask].repeat(N,1,1)
        
        self.densification_postfix(
            new_points=new_points,
            new_densities=new_densities, 
            new_scales=new_scales, 
            new_quaternions=new_quaternions,
            new_sh_coordinates_dc=new_sh_coordinates_dc, 
            new_sh_coordinates_rest=new_sh_coordinates_rest,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.model.device, dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_prune(self, max_grad:float=None, min_opacity:float=None, extent:float=None, max_screen_size:int=None):
        max_grad = self.max_grad if max_grad is None else max_grad
        min_opacity = self.min_opacity if min_opacity is None else min_opacity
        extent = self.scene_extent if extent is None else extent
        
        grads = self.points_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.model.strengths < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.model.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.model.strengths, torch.ones_like(self.model.all_densities.view(-1, 1))*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "all_densities")
        self.all_densities = optimizable_tensors["all_densities"]
