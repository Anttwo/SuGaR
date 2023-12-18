import bpy
import csv
import os
import json

# Get the path to the current Blender file
blend_file_path = bpy.data.filepath

# Get the directory of the Blender file
blend_dir = os.path.dirname(blend_file_path)

path = './'

# Construct the absolute path to the "data.csv" file
data_file_path = os.path.join(blend_dir, path)
data_file_path = os.path.join(data_file_path, "scene_tpose.json")  # TODO: Depends on the scene
print('Pose file path:', data_file_path)

res_dict = {}

obj = bpy.context.active_object
armature = obj.data

# Initialize dict
res_dict['matrix_world'] = [[obj.matrix_world[i][j] for j in range(4)] for i in range(4)]
res_dict['tpose_bones'] = {}
    
# Save rest poses for all bones
for bone in armature.bones:
    mat = obj.matrix_world @ bone.matrix_local
    mat_list = [[mat[i][j] for j in range(4)] for i in range(4)]
    res_dict['tpose_bones'][bone.name] = mat_list
        
with open(data_file_path, "w") as outfile:
    json.dump(res_dict, outfile)
    print(f'Results saved to "{data_file_path}".')
    