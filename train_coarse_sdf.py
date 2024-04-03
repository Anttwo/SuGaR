import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to optimize a coarse SuGaR model, i.e. a 3D Gaussian Splatting model with surface regularization losses in SDF space.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    parser.add_argument('-e', '--estimation_factor', type=float, default=0.2, help='factor to multiply the estimation loss by.')
    parser.add_argument('-n', '--normal_factor', type=float, default=0.2, help='factor to multiply the normal loss by.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    args = parser.parse_args()
    
    # Call function
    coarse_training_with_sdf_regularization(args)
    