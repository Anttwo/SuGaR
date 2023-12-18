import argparse
import os
import json
import open3d as o3d
import torch
from pytorch3d.io import load_objs_as_meshes
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.lpipsPyTorch import lpips
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB
from sugar_utils.general_utils import str2bool

from rich.console import Console
CONSOLE = Console(width=120)

os.makedirs('./lpipsPyTorch/weights/', exist_ok=True)
torch.hub.set_dir('./lpipsPyTorch/weights/')

n_skip_images_for_eval_split = 8


if __name__ == "__main__":
    
    # Parser
    parser = argparse.ArgumentParser(description='Script to extract a mesh from a coarse SuGaR scene.')
    
    # Config file for scenes to evaluate
    parser.add_argument('--scene_config', type=str, 
                        help='(Required) Path to the JSON file containing the scenes to evaluate. '
                        'The JSON file should be a dictionary with the following structure: '
                        '{source_images_dir_path: vanilla_gaussian_splatting_checkpoint_path}')
    
    # Coarse model parameters
    parser.add_argument('-i', '--iteration_to_load', type=int, default=7000, 
                        help='iteration to load.')
    parser.add_argument('-e', '--estimation_factor', type=float, default=0.2, 
                        help='Estimation factor to load for coarse model.')
    parser.add_argument('-r', '--regularization_type', type=str, 
                        help='(Required) Type of regularization to evaluate for coarse SuGaR. Can be "sdf" or "density".')
    
    # Mesh extraction parameters
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level used for mesh extraction in SuGaR models to evaluate. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in SuGaR meshes to evaluate. Default is 1_000_000')
    
    # Refined model parameters
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle used in the refined SuGaR models to evaluate. Default is 1')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations used in the refined SuGaR models to evaluate. Default is 15_000')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0, 
                        help='Index of GPU to use.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Evaluating standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Evaluating standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
    
    # (Optional) Additional evaluation parameters
    parser.add_argument('--evaluate_vanilla', type=str2bool, default=False, 
                        help='If True, will also evaluate vanilla 3DGS, in addition to SuGaR.')
    parser.add_argument('--use_uv_texture', type=str2bool, default=False, 
                        help='If True, will use the extracted UV texture for the mesh instead of Gaussian Splatting rendering.')
    parser.add_argument('--use_diffuse_color_only', type=str2bool, default=False, 
                        help='If True, will use only the diffuse component in Gaussian Splatting rendering.')
    parser.add_argument('--use_poisson_center', type=str2bool, default=False, 
                        help='If True, will evaluate Poisson center as the extraction method.')
    parser.add_argument('--use_marching_cubes', type=str2bool, default=False, 
                        help='If True, will evaluate Marching Cubes as the extraction method.')
    
    args = parser.parse_args()
    # Mesh resolution
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Evaluating low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Evaluating high poly config.')
    # Refinement time
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Evaluating short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Evaluating medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Evaluating long refinement time.')
            
    # --- Scenes dict ---
    with open(args.scene_config, 'r') as f:
        gs_checkpoints_eval = json.load(f)
    
    # --- Coarse model parameters ---
    coarse_iteration_to_load = args.iteration_to_load
    coarse_estimation_factor = args.estimation_factor
    estim_method = args.regularization_type
    coarse_normal_factor = 0.2
    
    # --- Mesh extraction parameters ---
    surface_levels = [args.surface_level]    
    decimation_targets = [args.n_vertices_in_mesh]
        
    # --- Refined model parameters ---
    surface_mesh_normal_consistency_factor = 0.1
    n_gaussians_per_surface_triangle_map = {args.n_vertices_in_mesh: args.gaussians_per_triangle,}
    refinement_iterations_list = [args.refinement_iterations]
        
    # --- Evaluation parameters ---
    evaluate_vanilla = args.evaluate_vanilla
    use_uv_texture = args.use_uv_texture
    use_diffuse_color_only = args.use_diffuse_color_only
    use_poisson_center = args.use_poisson_center
    use_marching_cubes = args.use_marching_cubes
            
    CONSOLE.print('==================================================')
    CONSOLE.print("Starting evaluation with the following parameters:")
    CONSOLE.print(f"Coarse iteration to load: {coarse_iteration_to_load}")
    CONSOLE.print(f"Coarse estimation factor: {coarse_estimation_factor}")
    CONSOLE.print(f"Coarse normal factor: {coarse_normal_factor}")
    CONSOLE.print(f"Estimation method: {estim_method}")
    CONSOLE.print(f"Surface levels: {surface_levels}")
    CONSOLE.print(f"Decimation targets: {decimation_targets}")
    CONSOLE.print(f"Surface mesh normal consistency factor: {surface_mesh_normal_consistency_factor}")
    CONSOLE.print(f"Number of Gaussians per surface triangle: {n_gaussians_per_surface_triangle_map}")
    CONSOLE.print(f"Refinement iterations: {refinement_iterations_list}")
    CONSOLE.print(f"GS checkpoints for evaluation: {gs_checkpoints_eval}")
    CONSOLE.print(f"Evaluate vanilla: {evaluate_vanilla}")
    CONSOLE.print(f"Use UV texture: {use_uv_texture}")
    CONSOLE.print(f"Use diffuse color only: {use_diffuse_color_only}")
    CONSOLE.print(f"Use Poisson center: {use_poisson_center}")
    CONSOLE.print(f"Use Marching Cubes: {use_marching_cubes}")
    CONSOLE.print('==================================================')
    
    # Set the GPU
    torch.cuda.set_device(args.gpu)
    
    # ==========================

    result_file_dir = './output/metrics/'
    os.makedirs(result_file_dir, exist_ok=True)
    results = {}
    
    for source_path in gs_checkpoints_eval.keys():
        scene_name = source_path.split('/')[-1]
        CONSOLE.print(f"\n===== Processing scene {scene_name}... =====")
        scene_results = {}
        
        # Loading vanilla 3DGS models
        gs_checkpoint_path = gs_checkpoints_eval[source_path]
        
        CONSOLE.print("Source path:", source_path)
        CONSOLE.print("Gaussian splatting checkpoint path:", gs_checkpoint_path)    
        CONSOLE.print(f"\nLoading Vanilla 3DGS model config {gs_checkpoint_path}...")
        
        nerfmodel_30k = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=gs_checkpoint_path,
            iteration_to_load=30_000,
            load_gt_images=True,
            eval_split=True,
            eval_split_interval=n_skip_images_for_eval_split,
            )

        nerfmodel_7k = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=gs_checkpoint_path,
            iteration_to_load=7000,
            load_gt_images=False,
            eval_split=True,
            eval_split_interval=n_skip_images_for_eval_split,
            )
        
        if use_diffuse_color_only:
            sh_deg_to_use = 0
        else:
            sh_deg_to_use = nerfmodel_30k.gaussians.active_sh_degree

        CONSOLE.print("Vanilla 3DGS Loaded.")
        CONSOLE.print("Number of test cameras:", len(nerfmodel_30k.test_cameras))
        CONSOLE.print("Using SH degree:", sh_deg_to_use)
        
        compute_lpips = True
        cam_indices = [cam_idx for cam_idx in range(len(nerfmodel_30k.test_cameras))]
        
        # Evaluating Vanilla 3DGS
        if evaluate_vanilla:
            CONSOLE.print("\n--- Starting Evaluation of Vanilla 3DGS... ---")

            gs_7k_ssims = []
            gs_7k_psnrs = []
            gs_7k_lpipss = []
            
            gs_30k_ssims = []
            gs_30k_psnrs = []
            gs_30k_lpipss = []
            
            with torch.no_grad():    
                for cam_idx in cam_indices:
                    # GT image
                    gt_img = nerfmodel_30k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                    
                    # Vanilla 3DGS image (30K)
                    gs_30k_img = nerfmodel_30k.render_image(
                        nerf_cameras=nerfmodel_30k.test_cameras,
                        camera_indices=cam_idx).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                    
                    # Vanilla 3DGS image (7K)
                    gs_7k_img = nerfmodel_7k.render_image(
                        nerf_cameras=nerfmodel_30k.test_cameras,
                        camera_indices=cam_idx).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                    
                    gs_30k_ssims.append(ssim(gs_30k_img, gt_img))
                    gs_30k_psnrs.append(psnr(gs_30k_img, gt_img))
                    gs_30k_lpipss.append(lpips(gs_30k_img, gt_img, net_type='vgg'))
                    
                    gs_7k_ssims.append(ssim(gs_7k_img, gt_img))
                    gs_7k_psnrs.append(psnr(gs_7k_img, gt_img))
                    gs_7k_lpipss.append(lpips(gs_7k_img, gt_img, net_type='vgg'))    
                    
            CONSOLE.print("Evaluation of Vanilla 3DGS finished.")
            scene_results['3dgs_7k'] = {}
            scene_results['3dgs_7k']['ssim'] = torch.tensor(gs_7k_ssims).mean().item()
            scene_results['3dgs_7k']['psnr'] = torch.tensor(gs_7k_psnrs).mean().item()
            scene_results['3dgs_7k']['lpips'] = torch.tensor(gs_7k_lpipss).mean().item()
            
            scene_results['3dgs_30k'] = {}
            scene_results['3dgs_30k']['ssim'] = torch.tensor(gs_30k_ssims).mean().item()
            scene_results['3dgs_30k']['psnr'] = torch.tensor(gs_30k_psnrs).mean().item()
            scene_results['3dgs_30k']['lpips'] = torch.tensor(gs_30k_lpipss).mean().item()
            
            CONSOLE.print(f"\nVanilla 3DGS results (7K iterations):")
            CONSOLE.print("SSIM:", torch.tensor(gs_7k_ssims).mean())
            CONSOLE.print("PSNR:", torch.tensor(gs_7k_psnrs).mean())
            CONSOLE.print("LPIPS:", torch.tensor(gs_7k_lpipss).mean())
            
            CONSOLE.print(f"\bVanilla 3DGS results (30K iterations):")
            CONSOLE.print("SSIM:", torch.tensor(gs_30k_ssims).mean())
            CONSOLE.print("PSNR:", torch.tensor(gs_30k_psnrs).mean())
            CONSOLE.print("LPIPS:", torch.tensor(gs_30k_lpipss).mean())
        
        # Evaluating SuGaR models
        with torch.no_grad():
            CONSOLE.print("\n--- Starting Evaluation of SuGaR... ---")
            for surface_level in surface_levels:
                for decimation_target in decimation_targets:
                    for refinement_iterations in refinement_iterations_list:
                        n_gaussians_per_surface_triangle = n_gaussians_per_surface_triangle_map[decimation_target]
                        
                        if use_uv_texture:
                            CONSOLE.print("Using UV texture for rendering.")
                            coarse_estimation_factor_str = str(coarse_estimation_factor).replace('.', '')
                            surface_level_str = str(surface_level).replace('.', '')
                            textured_mesh_path = os.path.join('./output/refined_mesh/', scene_name, 
                                        f'sugarfine_3Dgs{coarse_iteration_to_load}_{estim_method}estim{coarse_estimation_factor_str}_sdfnorm02_level{surface_level_str}_decim{decimation_target}_normalconsistency01_gaussperface{n_gaussians_per_surface_triangle}.obj')
                            CONSOLE.print(f'Loading textured mesh: {textured_mesh_path}')
                            
                            textured_mesh = load_objs_as_meshes([textured_mesh_path]).to(nerfmodel_30k.device)
                            CONSOLE.print(f"Loaded textured mesh with {len(textured_mesh.verts_list()[0])} vertices and {len(textured_mesh.faces_list()[0])} faces.")
                            
                            faces_per_pixel = 1
                            max_faces_per_bin = 50_000

                            from pytorch3d.renderer import (
                                AmbientLights,
                                RasterizationSettings, 
                                MeshRenderer, 
                                MeshRasterizer,  
                                SoftPhongShader,
                                )
                            from pytorch3d.renderer.blending import BlendParams

                            mesh_raster_settings = RasterizationSettings(
                                image_size=(nerfmodel_30k.image_height, nerfmodel_30k.image_width),
                                blur_radius=0.0, 
                                faces_per_pixel=faces_per_pixel,
                                # max_faces_per_bin=max_faces_per_bin
                            )
                            renderer = MeshRenderer(
                                rasterizer=MeshRasterizer(
                                    cameras=nerfmodel_30k.test_cameras.p3d_cameras[0], 
                                    raster_settings=mesh_raster_settings,
                                    ),
                                shader=SoftPhongShader(
                                    device=nerfmodel_30k.device, 
                                    cameras=nerfmodel_30k.test_cameras.p3d_cameras[0],
                                    lights=AmbientLights(device=nerfmodel_30k.device),
                                    blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
                                )
                            )
                            
                        else:
                            CONSOLE.print("Using surface Gaussian Splatting for rendering.")
                            sugar_checkpoint_path = f'sugarcoarse_3Dgs{coarse_iteration_to_load}_{estim_method}estimXX_sdfnormYY/'
                            sugar_checkpoint_path = sugar_checkpoint_path.replace(
                                'XX', str(coarse_estimation_factor).replace('.', '')
                                ).replace(
                                    'YY', str(coarse_normal_factor).replace('.', '')
                                    )
                            
                            # Loading mesh
                            CONSOLE.print(f"\nProcessing Surface level: {surface_level}, Decimation target: {decimation_target}, Refinement iterations: {refinement_iterations}...")
                            
                            if use_marching_cubes:
                                CONSOLE.print("Using Marching Cubes for mesh extraction.")
                                sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + 'marchingcubes_levelZZ_decimAA.ply'
                            elif use_poisson_center:
                                CONSOLE.print("Using Poisson Center for mesh extraction.")
                                sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_poissoncenters_decimAA.ply'
                            else:                        
                                sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_levelZZ_decimAA.ply'
                            sugar_mesh_path = sugar_mesh_path.replace(
                                'ZZ', str(surface_level).replace('.', '')
                                ).replace(
                                    'AA', str(decimation_target).replace('.', '')
                                    )
                            mesh_save_dir = os.path.join('./output/coarse_mesh/', scene_name)
                            sugar_mesh_path = os.path.join(mesh_save_dir, sugar_mesh_path)
                            CONSOLE.print(f'Loading mesh to bind to: {sugar_mesh_path}')
                            o3d_mesh = o3d.io.read_triangle_mesh(sugar_mesh_path)
                            
                            # Loading refined SuGaR model
                            mesh_name = sugar_mesh_path.split("/")[-1].split(".")[0]
                            refined_sugar_path = 'sugarfine_' + mesh_name.replace('sugarmesh_', '') + '_normalconsistencyXX_gaussperfaceYY/'
                            refined_sugar_path = os.path.join(os.path.join('./output/refined/', scene_name), refined_sugar_path)
                            refined_sugar_path = refined_sugar_path.replace(
                                'XX', str(surface_mesh_normal_consistency_factor).replace('.', '')
                                ).replace(
                                'YY', str(n_gaussians_per_surface_triangle).replace('.', '')
                                )
                            refined_sugar_path = os.path.join(refined_sugar_path, f'{refinement_iterations}.pt')
                            CONSOLE.print(f"Loading SuGaR model config {refined_sugar_path}...")
                            checkpoint = torch.load(refined_sugar_path, map_location=nerfmodel_30k.device)
                            refined_sugar = SuGaR(
                                nerfmodel=nerfmodel_30k,
                                points=checkpoint['state_dict']['_points'],
                                colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
                                initialize=False,
                                sh_levels=nerfmodel_30k.gaussians.active_sh_degree+1,
                                keep_track_of_knn=False,
                                knn_to_track=0,
                                beta_mode='average',
                                surface_mesh_to_bind=o3d_mesh,
                                n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
                                )
                            refined_sugar.load_state_dict(checkpoint['state_dict'])
                            refined_sugar.eval()
                        
                        # Evaluating SuGaR
                        with torch.no_grad():                
                            sugar_ssims = []
                            sugar_psnrs = []
                            sugar_lpipss = []
                            
                            for cam_idx in cam_indices:
                                # GT image
                                gt_img = nerfmodel_30k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                                
                                # SUGAR image
                                if use_uv_texture:
                                    p3d_cameras = nerfmodel_30k.test_cameras.p3d_cameras[cam_idx]
                                    sugar_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3].clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0).contiguous()
                                else:
                                    sugar_img = refined_sugar.render_image_gaussian_rasterizer(
                                        nerf_cameras=nerfmodel_30k.test_cameras,
                                        camera_indices=cam_idx,
                                        verbose=False,
                                        bg_color=None,
                                        sh_deg=sh_deg_to_use,
                                        compute_color_in_rasterizer=True,#compute_color_in_rasterizer,
                                    ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                                
                                sugar_ssims.append(ssim(sugar_img, gt_img))
                                sugar_psnrs.append(psnr(sugar_img, gt_img))
                                sugar_lpipss.append(lpips(sugar_img, gt_img, net_type='vgg'))
                        
                        CONSOLE.print(f"Evaluation of SuGaR finished, with surface level {surface_level} and decimation target {decimation_target} and refinement iterations {refinement_iterations}.")
                        str_surface_level = str(surface_level).replace('.', '')
                        scene_results[f'sugar_{str_surface_level}_{decimation_target}_{refinement_iterations}'] = {}
                        scene_results[f'sugar_{str_surface_level}_{decimation_target}_{refinement_iterations}']['ssim'] = torch.tensor(sugar_ssims).mean().item()
                        scene_results[f'sugar_{str_surface_level}_{decimation_target}_{refinement_iterations}']['psnr'] = torch.tensor(sugar_psnrs).mean().item()
                        scene_results[f'sugar_{str_surface_level}_{decimation_target}_{refinement_iterations}']['lpips'] = torch.tensor(sugar_lpipss).mean().item()
                        
                        CONSOLE.print(f"SuGaR results:")
                        CONSOLE.print("SSIM:", torch.tensor(sugar_ssims).mean())
                        CONSOLE.print("PSNR:", torch.tensor(sugar_psnrs).mean())
                        CONSOLE.print("LPIPS:", torch.tensor(sugar_lpipss).mean())
        
        # Saves results to JSON file                
        results[scene_name] = scene_results
        estim_factor_str = str(coarse_estimation_factor).replace('.', '')
        normal_factor_str = str(coarse_normal_factor).replace('.', '')
        if use_uv_texture:
            result_file_name = f'results_{estim_method}{estim_factor_str}_normal{normal_factor_str}_uvtexture.json'
        elif use_diffuse_color_only:
            result_file_name = f'results_{estim_method}{estim_factor_str}_normal{normal_factor_str}_diffuseonly.json'
        elif use_poisson_center:
            result_file_name = f'results_{estim_method}{estim_factor_str}_normal{normal_factor_str}_poissoncenter.json'
        elif use_marching_cubes:
            result_file_name = f'results_{estim_method}{estim_factor_str}_normal{normal_factor_str}_marchingcubes.json'
        else:
            result_file_name = f'results_{estim_method}{estim_factor_str}_normal{normal_factor_str}.json'
        result_file_name = os.path.join(result_file_dir, result_file_name)

        CONSOLE.print(f"Saving results to {result_file_name}...")
        with open(result_file_name, 'w') as f:
            json.dump(results, f, indent=4)