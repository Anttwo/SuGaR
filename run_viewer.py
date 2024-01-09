import sys
import os
import json
import argparse

viewer_path = './sugar_viewer'
json_path = os.path.join(viewer_path, 'src', 'scene_to_load.json')

if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to run the SuGaR viewer.')
    parser.add_argument('-p', '--ply_path',
                        type=str, 
                        default=None,
                        help='(Required) path to the refined SuGaR PLY file.')
    
    args = parser.parse_args()
    ply_path = args.ply_path
    
    # ----- Path checks -----
    if ply_path is None:
        raise ValueError('Please provide a path to the refined SuGaR PLY file.')
    if not os.path.exists(ply_path) or not ply_path.endswith('.ply'):
        raise ValueError('Could not find the refined SuGaR PLY file.')
    
    obj_path = ply_path.replace('.ply', '.obj').replace('refined_ply', 'refined_mesh')
    png_path = obj_path.replace('.obj', '.png')
    
    if not os.path.exists(obj_path):
        raise ValueError('Could not find the corresponding refined SuGaR OBJ file.')
    else:
        print('\nFound a matching refined SuGaR OBJ file.')
    
    if not os.path.exists(png_path):
        raise ValueError('Could not find the corresponding refined SuGaR texture PNG file.')
    else:
        print('Found a matching refined SuGaR texture PNG file.\n')
        
    # ----- Write the paths to a json file -----
    with open(json_path, 'w') as f:
        json.dump({
            'ply_path': ply_path, 
            'obj_path': obj_path, 
            'png_path': png_path}, f)
    
    # ----- Run the viewer -----
    os.system(f'npm run --prefix {viewer_path} dev')
    