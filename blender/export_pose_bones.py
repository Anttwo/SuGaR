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
data_file_path = os.path.join(data_file_path, "animation.json")  # TODO: Depends on the animation
print('Pose file path:', data_file_path)

start_frame = 1
end_frame = 100 # TODO: Depends on the animation

res_dict = {}

obj = bpy.context.active_object
armature = obj.data

# Initialize dict
res_dict['matrix_world'] = [[obj.matrix_world[i][j] for j in range(4)] for i in range(4)]
res_dict['rest_bones'] = {}
res_dict['pose_bones'] = {}
    
# Save rest poses for all bones
for bone in armature.bones:
    mat = obj.matrix_world @ bone.matrix_local
    mat_list = [[mat[i][j] for j in range(4)] for i in range(4)]
    res_dict['rest_bones'][bone.name] = mat_list
    res_dict['pose_bones'][bone.name] = []

# Save frame poses for all bones
for i_frame in range(start_frame, end_frame+1):
    print('Frame', i_frame)
    
    # Set frame
    bpy.context.scene.frame_set(i_frame)
    
    # Get object transform
    obj = bpy.context.active_object
    
    pose = obj.pose
    armature = obj.data
    
    print("pose_position:", armature.pose_position)
    
    for bone in pose.bones:
        mat = obj.matrix_world @ bone.matrix
        mat_list = [[mat[i][j] for j in range(4)] for i in range(4)]
        res_dict['pose_bones'][bone.name].append(mat_list)
        
with open(data_file_path, "w") as outfile:
    json.dump(res_dict, outfile)
    print(f'Results saved to "{data_file_path}".')
    