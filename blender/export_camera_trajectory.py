import bpy
import csv
import os
import json

# Get the path to the current Blender file
blend_file_path = bpy.data.filepath

# Get the directory of the Blender file
blend_dir = os.path.dirname(blend_file_path)

# Construct the absolute path to the "data.csv" file
path = './'
data_file_path = os.path.join(blend_dir, path)
data_file_path = os.path.join(data_file_path, "camera_trajectory.json")
print('Camera poses file path:', data_file_path)

res_dict = {}
res_dict['matrix_world'] = []

start_frame = 1
end_frame = 71

# Save frame poses for all bones
for i_frame in range(start_frame, end_frame+1):
    print('\nFrame', i_frame)
    
    # Set frame
    bpy.context.scene.frame_set(i_frame)
    
    # Save camera pose
    obj = bpy.context.active_object
    res_dict['matrix_world'].append([[obj.matrix_world[i][j] for j in range(4)] for i in range(4)])
    
    print('matrix world:')
    print(res_dict['matrix_world'])
    
        
with open(data_file_path, "w") as outfile:
    json.dump(res_dict, outfile)
    print(f'Results saved to "{data_file_path}".')
    