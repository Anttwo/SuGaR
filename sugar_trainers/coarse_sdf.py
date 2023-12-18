import os
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from sugar_scene.gs_model import GaussianSplattingWrapper, fetchPly
from sugar_scene.sugar_model import SuGaR
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_scene.sugar_densifier import SuGaRDensifier
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss

from rich.console import Console
import time


def coarse_training_with_sdf_regularization(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    use_eval_split = True
    n_skip_images_for_eval_split = 8

    freeze_gaussians = False
    initialize_from_trained_3dgs = True  # True or False
    if initialize_from_trained_3dgs:
        prune_at_start = False
        start_pruning_threshold = 0.5
    no_rendering = freeze_gaussians

    n_points_at_start = None  # If None, takes all points in the SfM point cloud

    learnable_positions = True  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
    sh_levels = 4  

        
    # -----Radiance Mesh-----
    triangle_scale=1.
    
        
    # -----Rendering parameters-----
    compute_color_in_rasterizer = False  # TODO: Try True

        
    # -----Optimization parameters-----

    # Learning rates and scheduling
    num_iterations = 15_000  # Changed

    spatial_lr_scale = None
    position_lr_init=0.00016
    position_lr_final=0.0000016
    position_lr_delay_mult=0.01
    position_lr_max_steps=30_000
    feature_lr=0.0025
    opacity_lr=0.05
    scaling_lr=0.005
    rotation_lr=0.001
        
    # Densifier and pruning
    heavy_densification = False
    if initialize_from_trained_3dgs:
        densify_from_iter = 500 + 99999 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 - 7000 # 7000
    else:
        densify_from_iter = 500 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 # 7000

    if heavy_densification:
        densification_interval = 50  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0001  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01
    else:
        densification_interval = 100  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0002  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01

    # Data processing and batching
    n_images_to_use_for_training = -1  # If -1, uses all images

    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2

    # Regularization
    enforce_entropy_regularization = True
    if enforce_entropy_regularization:
        start_entropy_regularization_from = 7000
        end_entropy_regularization_at = 9000  # TODO: Change
        entropy_regularization_factor = 0.1
            
    regularize_sdf = True
    if regularize_sdf:
        beta_mode = 'average'  # 'learnable', 'average' or 'weighted_average'
        
        start_sdf_regularization_from = 9000
        regularize_sdf_only_for_gaussians_with_high_opacity = False
        if regularize_sdf_only_for_gaussians_with_high_opacity:
            sdf_regularization_opacity_threshold = 0.5
            
        use_sdf_estimation_loss = True
        enforce_samples_to_be_on_surface = False
        if use_sdf_estimation_loss or enforce_samples_to_be_on_surface:
            sdf_estimation_mode = 'sdf'  # 'sdf' or 'density'
            # sdf_estimation_factor = 0.2  # 0.1 or 0.2?
            samples_on_surface_factor = 0.2  # 0.05
            
            squared_sdf_estimation_loss = False
            squared_samples_on_surface_loss = False
            
            normalize_by_sdf_std = False  # False
            
            start_sdf_estimation_from = 9000  # 7000
            
            sample_only_in_gaussians_close_to_surface = True
            close_gaussian_threshold = 2.  # 2.
            
            backpropagate_gradients_through_depth = True  # True
            
        use_sdf_better_normal_loss = True
        if use_sdf_better_normal_loss:
            start_sdf_better_normal_from = 9000
            # sdf_better_normal_factor = 0.2  # 0.1 or 0.2?
            sdf_better_normal_gradient_through_normal_only = True
        
        density_factor = 1. / 16. # 1. / 16.
        if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and sdf_estimation_mode == 'density':
            density_factor = 1.
        density_threshold = 1.  # 0.5 * density_factor
        n_samples_for_sdf_regularization = 1_000_000  # 300_000
        sdf_sampling_scale_factor = 1.5
        sdf_sampling_proportional_to_volume = False
        
    bind_to_surface_mesh = False
    if bind_to_surface_mesh:
        learn_surface_mesh_positions = True
        learn_surface_mesh_opacity = True
        learn_surface_mesh_scales = True
        n_gaussians_per_surface_triangle=6  # 1, 3, 4 or 6
        
        use_surface_mesh_laplacian_smoothing_loss = True
        if use_surface_mesh_laplacian_smoothing_loss:
            surface_mesh_laplacian_smoothing_method = "uniform"  # "cotcurv", "cot", "uniform"
            surface_mesh_laplacian_smoothing_factor = 5.  # 0.1
        
        use_surface_mesh_normal_consistency_loss = True
        if use_surface_mesh_normal_consistency_loss:
            surface_mesh_normal_consistency_factor = 0.1  # 0.1
            
        densify_from_iter = 999_999
        densify_until_iter = 0
        position_lr_init=0.00016 * 0.01
        position_lr_final=0.0000016 * 0.01
        scaling_lr=0.005
    else:
        surface_mesh_to_bind_path = None
        
    if regularize_sdf:
        regularize = True
        regularity_knn = 16  # 8 until now
        # regularity_knn = 8
        regularity_samples = -1 # Retry with 1000, 10000
        reset_neighbors_every = 500  # 500 until now
        regularize_from = 7000  # 0 until now
        start_reset_neighbors_from = 7000+1  # 0 until now (should be equal to regularize_from + 1?)
        prune_when_starting_regularization = False
    else:
        regularize = False
        regularity_knn = 0
    if bind_to_surface_mesh:
        regularize = False
        regularity_knn = 0
        
    # Opacity management
    prune_low_opacity_gaussians_at = [9000]
    if bind_to_surface_mesh:
        prune_low_opacity_gaussians_at = [999_999]
    prune_hard_opacity_threshold = 0.5

    # Warmup
    do_resolution_warmup = False
    if do_resolution_warmup:
        resolution_warmup_every = 500
        current_resolution_factor = downscale_resolution_factor * 4.
    else:
        current_resolution_factor = downscale_resolution_factor

    do_sh_warmup = True  # Should be True
    if initialize_from_trained_3dgs:
        do_sh_warmup = False
        sh_levels = 4  # nerfmodel.gaussians.active_sh_degree + 1
        CONSOLE.print("Changing sh_levels to match the loaded model:", sh_levels)
    if do_sh_warmup:
        sh_warmup_every = 1000
        current_sh_levels = 1
    else:
        current_sh_levels = sh_levels
        

    # -----Log and save-----
    print_loss_every_n_iterations = 50
    save_model_every_n_iterations = 1_000_000
    save_milestones = [9000, 12_000, 15_000]

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-2])
            
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load    
    
    sdf_estimation_factor = args.estimation_factor
    sdf_better_normal_factor = args.normal_factor
    
    sugar_checkpoint_path = f'sugarcoarse_3Dgs{iteration_to_load}_sdfestimXX_sdfnormYY/'
    sugar_checkpoint_path = os.path.join(args.output_dir, sugar_checkpoint_path)
    sugar_checkpoint_path = sugar_checkpoint_path.replace(
        'XX', str(sdf_estimation_factor).replace('.', '')
        ).replace(
            'YY', str(sdf_better_normal_factor).replace('.', '')
            )
    
    use_eval_split = args.eval
    
    ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
    
    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("SUGAR checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Iteration to load:", iteration_to_load)
    CONSOLE.print("Output directory:", args.output_dir)
    CONSOLE.print("SDF estimation factor:", sdf_estimation_factor)
    CONSOLE.print("SDF better normal factor:", sdf_better_normal_factor)
    CONSOLE.print("Eval split:", use_eval_split)
    CONSOLE.print("---------------------------")
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(detect_anomaly)
    
    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
       nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )

    # Point cloud
    if initialize_from_trained_3dgs:
        with torch.no_grad():    
            print("Initializing model from trained 3DGS...")
            with torch.no_grad():
                sh_levels = int(np.sqrt(nerfmodel.gaussians.get_features.shape[1]))
            
            from sugar_utils.spherical_harmonics import SH2RGB
            points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
            colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
            if prune_at_start:
                with torch.no_grad():
                    start_prune_mask = nerfmodel.gaussians.get_opacity.view(-1) > start_pruning_threshold
                    points = points[start_prune_mask]
                    colors = colors[start_prune_mask]
            n_points = len(points)
    else:
        CONSOLE.print("\nLoading SfM point cloud...")
        pcd = fetchPly(ply_path)
        points = torch.tensor(pcd.points, device=nerfmodel.device).float().cuda()
        colors = torch.tensor(pcd.colors, device=nerfmodel.device).float().cuda()
    
        if n_points_at_start is not None:
            n_points = n_points_at_start
            pts_idx = torch.randperm(len(points))[:n_points]
            points, colors = points.to(device)[pts_idx], colors.to(device)[pts_idx]
        else:
            n_points = len(points)
            
    CONSOLE.print(f"Point cloud generated. Number of points: {len(points)}")
    
    # Mesh to bind to if needed  TODO
    if bind_to_surface_mesh:
        surface_mesh_to_bind_full_path = os.path.join('./results/meshes/', surface_mesh_to_bind_path)
        CONSOLE.print(f'\nLoading mesh to bind to: {surface_mesh_to_bind_full_path}...')
        o3d_mesh = o3d.io.read_triangle_mesh(surface_mesh_to_bind_full_path)
        CONSOLE.print("Mesh to bind to loaded.")
    else:
        o3d_mesh = None
        learn_surface_mesh_positions = False
        learn_surface_mesh_opacity = False
        learn_surface_mesh_scales = False
        n_gaussians_per_surface_triangle=1
    
    if not regularize_sdf:
        beta_mode = None
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=points, #nerfmodel.gaussians.get_xyz.data,
        colors=colors, #0.5 + _C0 * nerfmodel.gaussians.get_features.data[:, 0, :],
        initialize=True,
        sh_levels=sh_levels,
        learnable_positions=learnable_positions,
        triangle_scale=triangle_scale,
        keep_track_of_knn=regularize,
        knn_to_track=regularity_knn,
        beta_mode=beta_mode,
        freeze_gaussians=freeze_gaussians,
        surface_mesh_to_bind=o3d_mesh,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=learn_surface_mesh_positions,
        learn_surface_mesh_opacity=learn_surface_mesh_opacity,
        learn_surface_mesh_scales=learn_surface_mesh_scales,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )
    
    if initialize_from_trained_3dgs:
        with torch.no_grad():            
            CONSOLE.print("Initializing 3D gaussians from 3D gaussians...")
            if prune_at_start:
                sugar._scales[...] = nerfmodel.gaussians._scaling.detach()[start_prune_mask]
                sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()[start_prune_mask]
                sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()[start_prune_mask]
                sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()[start_prune_mask]
                sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()[start_prune_mask]
            else:
                sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
                sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
                sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
                sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
                sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
        
    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')
    
    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)
 
    torch.cuda.empty_cache()
    
    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()
    
    
    # ====================Initialize optimizer====================
    if spatial_lr_scale is None:
        spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)
    
    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])
        
        
    # ====================Initialize densifier====================
    gaussian_densifier = SuGaRDensifier(
        sugar_model=sugar,
        sugar_optimizer=optimizer,
        max_grad=densify_grad_threshold,
        min_opacity=prune_opacity_threshold,
        max_screen_size=densify_screen_size_threshold,
        scene_extent=cameras_spatial_extent,
        percent_dense=densification_percent_distinction,
        )
    CONSOLE.print("Densifier initialized.")
        
    
    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')
    
    
    # ====================Start training====================
    sugar.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()
    
    if initialize_from_trained_3dgs:
        iteration = 7000 - 1
    
    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break
        
        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)
        
        # We iterate on images
        for i in range(0, train_num_images, train_num_images_per_batch):
            iteration += 1
            
            # Update learning rates
            optimizer.update_learning_rate(iteration)
            
            # Prune low-opacity gaussians for optimizing triangles
            if (
                regularize and prune_when_starting_regularization and iteration == regularize_from + 1
                ) or (
                (iteration-1) in prune_low_opacity_gaussians_at
                ):
                CONSOLE.print("\nPruning gaussians with low-opacity for further optimization...")
                prune_mask = (gaussian_densifier.model.strengths < prune_hard_opacity_threshold).squeeze()
                gaussian_densifier.prune_points(prune_mask)
                CONSOLE.print(f'Pruning finished: {sugar.n_points} gaussians left.')
                if regularize and iteration >= start_reset_neighbors_from:
                    sugar.reset_neighbors()
            
            start_idx = i
            end_idx = min(i+train_num_images_per_batch, train_num_images)
            
            camera_indices = shuffled_idx[start_idx:end_idx]
            
            # Computing rgb predictions
            if not no_rendering:
                outputs = sugar.render_image_gaussian_rasterizer( 
                    camera_indices=camera_indices.item(),
                    verbose=False,
                    bg_color = None,
                    sh_deg=current_sh_levels-1,
                    sh_rotations=None,
                    compute_color_in_rasterizer=compute_color_in_rasterizer,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                    return_opacities=enforce_entropy_regularization,
                    )
                pred_rgb = outputs['image'].view(-1, 
                    sugar.image_height, 
                    sugar.image_width, 
                    3)
                radii = outputs['radii']
                viewspace_points = outputs['viewspace_points']
                if enforce_entropy_regularization:
                    opacities = outputs['opacities']
                
                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute
                
                # Gather rgb ground truth
                gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)           
                gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                    
                # Compute loss 
                loss = loss_fn(pred_rgb, gt_rgb)
                        
                if enforce_entropy_regularization and iteration > start_entropy_regularization_from and iteration < end_entropy_regularization_at:
                    if iteration == start_entropy_regularization_from + 1:
                        CONSOLE.print("\n---INFO---\nStarting entropy regularization.")
                    if iteration == end_entropy_regularization_at - 1:
                        CONSOLE.print("\n---INFO---\nStopping entropy regularization.")
                    visibility_filter = radii > 0
                    if visibility_filter is not None:
                        vis_opacities = opacities[visibility_filter]
                    else:
                        vis_opacities = opacities
                    loss = loss + entropy_regularization_factor * (
                        - vis_opacities * torch.log(vis_opacities + 1e-10)
                        - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                        ).mean()
                
                if regularize:
                    if iteration == regularize_from:
                        CONSOLE.print("Starting regularization...")
                        # sugar.reset_neighbors()
                    if iteration > regularize_from:
                        visibility_filter = radii > 0
                        if (iteration >= start_reset_neighbors_from) and ((iteration == regularize_from + 1) or (iteration % reset_neighbors_every == 0)):
                            CONSOLE.print("\n---INFO---\nResetting neighbors...")
                            sugar.reset_neighbors()
                        neighbor_idx = sugar.get_neighbors_of_random_points(num_samples=regularity_samples,)
                        if visibility_filter is not None:
                            neighbor_idx = neighbor_idx[visibility_filter]  # TODO: Error here

                        if regularize_sdf and iteration > start_sdf_regularization_from:
                            if iteration == start_sdf_regularization_from + 1:
                                CONSOLE.print("\n---INFO---\nStarting SDF regularization.")
                            
                            sampling_mask = visibility_filter
                            
                            if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and iteration > start_sdf_estimation_from:
                                if iteration == start_sdf_estimation_from + 1:
                                    CONSOLE.print("\n---INFO---\nStarting SDF estimation loss.")
                                fov_camera = nerfmodel.training_cameras.p3d_cameras[camera_indices.item()]
                                
                                # Render a depth map using gaussian splatting
                                if backpropagate_gradients_through_depth:                                
                                    point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
                                    max_depth = point_depth.max()
                                    depth = sugar.render_image_gaussian_rasterizer(
                                                camera_indices=camera_indices.item(),
                                                bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=sugar.device),
                                                sh_deg=0,
                                                compute_color_in_rasterizer=False,#compute_color_in_rasterizer,
                                                compute_covariance_in_rasterizer=True,
                                                return_2d_radii=False,
                                                use_same_scale_in_all_directions=False,
                                                point_colors=point_depth,
                                            )[..., 0]
                                else:
                                    with torch.no_grad():
                                        point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
                                        max_depth = point_depth.max()
                                        depth = sugar.render_image_gaussian_rasterizer(
                                                    camera_indices=camera_indices.item(),
                                                    bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=sugar.device),
                                                    sh_deg=0,
                                                    compute_color_in_rasterizer=False,#compute_color_in_rasterizer,
                                                    compute_covariance_in_rasterizer=True,
                                                    return_2d_radii=False,
                                                    use_same_scale_in_all_directions=False,
                                                    point_colors=point_depth,
                                                )[..., 0]
                                
                                # If needed, compute which gaussians are close to the surface in the depth map.
                                # Then, we sample points only in these gaussians.
                                # TODO: Compute projections only for gaussians in visibility filter.
                                # TODO: Is the torch.no_grad() a good idea?
                                if sample_only_in_gaussians_close_to_surface:
                                    with torch.no_grad():
                                        gaussian_to_camera = torch.nn.functional.normalize(fov_camera.get_camera_center() - sugar.points, dim=-1)
                                        gaussian_centers_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(sugar.points)
                                        
                                        gaussian_centers_z = gaussian_centers_in_camera_space[..., 2] + 0.
                                        gaussian_centers_map_z = sugar.get_points_depth_in_depth_map(fov_camera, depth, gaussian_centers_in_camera_space)
                                        
                                        gaussian_standard_deviations = (
                                            sugar.scaling * quaternion_apply(quaternion_invert(sugar.quaternions), gaussian_to_camera)
                                            ).norm(dim=-1)
                                    
                                        gaussians_close_to_surface = (gaussian_centers_map_z - gaussian_centers_z).abs() < close_gaussian_threshold * gaussian_standard_deviations
                                        sampling_mask = sampling_mask * gaussians_close_to_surface
                            
                            n_gaussians_in_sampling = sampling_mask.sum()
                            if n_gaussians_in_sampling > 0:
                                sdf_samples, sdf_gaussian_idx = sugar.sample_points_in_gaussians(
                                    num_samples=n_samples_for_sdf_regularization, 
                                    sampling_scale_factor=sdf_sampling_scale_factor,
                                    mask=sampling_mask,
                                    probabilities_proportional_to_volume=sdf_sampling_proportional_to_volume,
                                    )
                                
                                if use_sdf_estimation_loss or use_sdf_better_normal_loss:
                                    fields = sugar.get_field_values(
                                        sdf_samples, sdf_gaussian_idx, 
                                        return_sdf=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (sdf_estimation_mode=='sdf') and iteration > start_sdf_estimation_from, 
                                        density_threshold=density_threshold, density_factor=density_factor, 
                                        return_sdf_grad=False, sdf_grad_max_value=10.,
                                        return_closest_gaussian_opacities=use_sdf_better_normal_loss and iteration > start_sdf_better_normal_from,
                                        return_beta=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (sdf_estimation_mode=='density') and iteration > start_sdf_estimation_from,
                                        )
                                
                                if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and iteration > start_sdf_estimation_from:
                                    # Compute the depth of the points in the gaussians
                                    sdf_samples_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(sdf_samples)
                                    sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.
                                    proj_mask = sdf_samples_z > fov_camera.znear
                                    sdf_samples_map_z = sugar.get_points_depth_in_depth_map(fov_camera, depth, sdf_samples_in_camera_space[proj_mask])
                                    sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]
                                    
                                    if not sample_only_in_gaussians_close_to_surface:
                                        raise NotImplementedError("Not implemented yet.")
                                    
                                    with torch.no_grad():
                                        if normalize_by_sdf_std:
                                            sdf_sample_std = gaussian_standard_deviations[sdf_gaussian_idx][proj_mask]
                                        else:
                                            sdf_sample_std = sugar.get_cameras_spatial_extent() / 10.
                                    
                                    if use_sdf_estimation_loss:
                                        if sdf_estimation_mode == 'sdf':
                                            sdf_values = fields['sdf'][proj_mask]
                                            if squared_sdf_estimation_loss:
                                                sdf_estimation_loss = ((sdf_values - sdf_estimation.abs()) / sdf_sample_std).pow(2)
                                            else:
                                                sdf_estimation_loss = (sdf_values - sdf_estimation.abs()).abs() / sdf_sample_std
                                            loss = loss + sdf_estimation_factor * sdf_estimation_loss.clamp(max=10.*sugar.get_cameras_spatial_extent()).mean()
                                        elif sdf_estimation_mode == 'density':
                                            beta = fields['beta'][proj_mask]
                                            densities = fields['density'][proj_mask]
                                            target_densities = torch.exp(-0.5 * sdf_estimation.pow(2) / beta.pow(2))
                                            if squared_sdf_estimation_loss:
                                                sdf_estimation_loss = ((densities - target_densities)).pow(2)
                                            else:
                                                sdf_estimation_loss = (densities - target_densities).abs()
                                            loss = loss + sdf_estimation_factor * sdf_estimation_loss.mean()
                                        else:
                                            raise ValueError(f"Unknown sdf_estimation_mode: {sdf_estimation_mode}")

                                    if enforce_samples_to_be_on_surface:
                                        if squared_samples_on_surface_loss:
                                            samples_on_surface_loss = (sdf_estimation / sdf_sample_std).pow(2)
                                        else:
                                            samples_on_surface_loss = sdf_estimation.abs() / sdf_sample_std
                                        loss = loss + samples_on_surface_factor * samples_on_surface_loss.clamp(max=10.*sugar.get_cameras_spatial_extent()).mean()
                                        
                                if use_sdf_better_normal_loss and (iteration > start_sdf_better_normal_from):
                                    if iteration == start_sdf_better_normal_from + 1:
                                        CONSOLE.print("\n---INFO---\nStarting SDF better normal loss.")
                                    closest_gaussians_idx = sugar.knn_idx[sdf_gaussian_idx]
                                    # Compute minimum scaling
                                    closest_min_scaling = sugar.scaling.min(dim=-1)[0][closest_gaussians_idx].detach().view(len(sdf_samples), -1)
                                    
                                    # Compute normals and flip their sign if needed
                                    closest_gaussian_normals = sugar.get_normals(estimate_from_points=False)[closest_gaussians_idx]
                                    samples_gaussian_normals = sugar.get_normals(estimate_from_points=False)[sdf_gaussian_idx]
                                    closest_gaussian_normals = closest_gaussian_normals * torch.sign(
                                        (closest_gaussian_normals * samples_gaussian_normals[:, None]).sum(dim=-1, keepdim=True)
                                        ).detach()
                                    
                                    # Compute weights for normal regularization, based on the gradient of the sdf
                                    closest_gaussian_opacities = fields['closest_gaussian_opacities'].detach()  # Shape is (n_samples, n_neighbors)
                                    normal_weights = ((sdf_samples[:, None] - sugar.points[closest_gaussians_idx]) * closest_gaussian_normals).sum(dim=-1).abs()  # Shape is (n_samples, n_neighbors)
                                    if sdf_better_normal_gradient_through_normal_only:
                                        normal_weights = normal_weights.detach()
                                    normal_weights =  closest_gaussian_opacities * normal_weights / closest_min_scaling.clamp(min=1e-6)**2  # Shape is (n_samples, n_neighbors)
                                    
                                    # The weights should have a sum of 1 because of the eikonal constraint
                                    normal_weights_sum = normal_weights.sum(dim=-1).detach()  # Shape is (n_samples,)
                                    normal_weights = normal_weights / normal_weights_sum.unsqueeze(-1).clamp(min=1e-6)  # Shape is (n_samples, n_neighbors)
                                    
                                    # Compute regularization loss
                                    sdf_better_normal_loss = (samples_gaussian_normals - (normal_weights[..., None] * closest_gaussian_normals).sum(dim=-2)
                                                              ).pow(2).sum(dim=-1)  # Shape is (n_samples,)
                                    loss = loss + sdf_better_normal_factor * sdf_better_normal_loss.mean()
                            else:
                                CONSOLE.log("WARNING: No gaussians available for sampling.")
                                
            else:
                loss = 0.
                
            # Surface mesh optimization
            if bind_to_surface_mesh:
                surface_mesh = sugar.surface_mesh
                
                if use_surface_mesh_laplacian_smoothing_loss:
                    loss = loss + surface_mesh_laplacian_smoothing_factor * mesh_laplacian_smoothing(
                        surface_mesh, method=surface_mesh_laplacian_smoothing_method)
                
                if use_surface_mesh_normal_consistency_loss:
                    loss = loss + surface_mesh_normal_consistency_factor * mesh_normal_consistency(surface_mesh)
            
            # Update parameters
            loss.backward()
            
            # Densification
            with torch.no_grad():
                if (not no_rendering) and (iteration < densify_until_iter):
                    gaussian_densifier.update_densification_stats(viewspace_points, radii, visibility_filter=radii>0)

                    if iteration > densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = gaussian_densifier.max_screen_size if iteration > opacity_reset_interval else None
                        gaussian_densifier.densify_and_prune(densify_grad_threshold, prune_opacity_threshold, 
                                                    cameras_spatial_extent, size_threshold)
                        CONSOLE.print("Gaussians densified and pruned. New number of gaussians:", len(sugar.points))
                        
                        if regularize and (iteration > regularize_from) and (iteration >= start_reset_neighbors_from):
                            sugar.reset_neighbors()
                            CONSOLE.print("Neighbors reset.")
                    
                    if iteration % opacity_reset_interval == 0:
                        gaussian_densifier.reset_opacity()
                        CONSOLE.print("Opacity reset.")
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            # Print loss
            if iteration==1 or iteration % print_loss_every_n_iterations == 0:
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                    "computed in", (time.time() - t0) / 60., "minutes.")
                with torch.no_grad():
                    scales = sugar.scaling.detach()
             
                    CONSOLE.print("------Stats-----")
                    CONSOLE.print("---Min, Max, Mean, Std")
                    CONSOLE.print("Points:", sugar.points.min().item(), sugar.points.max().item(), sugar.points.mean().item(), sugar.points.std().item(), sep='   ')
                    CONSOLE.print("Scaling factors:", sugar.scaling.min().item(), sugar.scaling.max().item(), sugar.scaling.mean().item(), sugar.scaling.std().item(), sep='   ')
                    CONSOLE.print("Quaternions:", sugar.quaternions.min().item(), sugar.quaternions.max().item(), sugar.quaternions.mean().item(), sugar.quaternions.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates dc:", sugar._sh_coordinates_dc.min().item(), sugar._sh_coordinates_dc.max().item(), sugar._sh_coordinates_dc.mean().item(), sugar._sh_coordinates_dc.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates rest:", sugar._sh_coordinates_rest.min().item(), sugar._sh_coordinates_rest.max().item(), sugar._sh_coordinates_rest.mean().item(), sugar._sh_coordinates_rest.std().item(), sep='   ')
                    CONSOLE.print("Opacities:", sugar.strengths.min().item(), sugar.strengths.max().item(), sugar.strengths.mean().item(), sugar.strengths.std().item(), sep='   ')
                    if regularize_sdf and iteration > start_sdf_regularization_from:
                        CONSOLE.print("Number of gaussians used for sampling in SDF regularization:", n_gaussians_in_sampling)
                t0 = time.time()
                
            # Save model
            if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                CONSOLE.print("Saving model...")
                model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                sugar.save_model(path=model_path,
                                train_losses=train_losses,
                                epoch=epoch,
                                iteration=iteration,
                                optimizer_state_dict=optimizer.state_dict(),
                                )
                # if optimize_triangles and iteration >= optimize_triangles_from:
                #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                CONSOLE.print("Model saved.")
            
            if iteration >= num_iterations:
                break
            
            if do_sh_warmup and (iteration > 0) and (current_sh_levels < sh_levels) and (iteration % sh_warmup_every == 0):
                current_sh_levels += 1
                CONSOLE.print("Increasing number of spherical harmonics levels to", current_sh_levels)
            
            if do_resolution_warmup and (iteration > 0) and (current_resolution_factor > 1) and (iteration % resolution_warmup_every == 0):
                current_resolution_factor /= 2.
                nerfmodel.downscale_output_resolution(1/2)
                CONSOLE.print(f'\nCamera resolution scaled to '
                        f'{nerfmodel.training_cameras.ns_cameras.height[0].item()} x '
                        f'{nerfmodel.training_cameras.ns_cameras.width[0].item()}'
                        )
                sugar.adapt_to_cameras(nerfmodel.training_cameras)
                # TODO: resize GT images
        
        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    optimizer_state_dict=optimizer.state_dict(),
                    )

    CONSOLE.print("Final model saved.")
    return model_path