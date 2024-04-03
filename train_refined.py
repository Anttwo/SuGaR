import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.refine import refined_training

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to refine a SuGaR model.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')  
    parser.add_argument('-m', '--mesh_path', 
                        type=str, 
                        help='Path to the extracted mesh file to use for refinement.')  
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')  
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')  
    
    parser.add_argument('-n', '--normal_consistency_factor', type=float, default=0.1, 
                        help='Factor to multiply the normal consistency loss by.')  
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1, 
                        help='Number of gaussians per triangle.')  
    parser.add_argument('-v', '--n_vertices_in_fg', type=int, default=1_000_000, 
                        help='Number of vertices in the foreground (Mesh resolution). Used for computing learning rates.')  
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')  
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    
    parser.add_argument('--export_ply', type=str2bool, default=True, 
                        help='If True, export a ply files with the refined 3D Gaussians at the end of the training.')

    args = parser.parse_args()
    
    # Call function
    refined_training(args)
    