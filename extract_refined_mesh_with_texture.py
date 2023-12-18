import argparse
from sugar_utils.general_utils import str2bool
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to train a full macarons model in large 3D scenes.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  # --OK
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')  # --OK
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')  # --OK
    parser.add_argument('-m', '--refined_model_path',
                        type=str, 
                        help='(Required) Path to the refine model checkpoint.')  # --OK
    parser.add_argument('-o', '--mesh_output_dir',
                        type=str, 
                        default=None, 
                        help='path to the output directory.')  # --OK
    parser.add_argument('-n', '--n_gaussians_per_surface_triangle',
                        default=None, type=int, help='Number of gaussians per surface triangle.')  # --OK
    parser.add_argument('--square_size',
                        default=None, type=int, help='Size of the square to use for the texture.')  # --OK
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Index of GPU to use.')
    
    # Optional postprocessing
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False, 
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                        'This step takes a few minutes and is not needed in general, as it can also be risky. '
                        'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')  # --OK
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')  # --OK
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')  # --OK
    
    args = parser.parse_args()
    
    # Call function
    extract_mesh_and_texture_from_refined_sugar(args)
    