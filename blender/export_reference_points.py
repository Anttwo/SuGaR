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
data_file_path = os.path.join(data_file_path, "scene_reference_points.json")  # TODO: Depends on the scene
print('Points file path:', data_file_path)

# Get data
obj = bpy.context.active_object
m = bpy.context.object.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
vertices = m.vertices        
print("Number of vertices:", len(m.vertices))

# Initialize dict
res_dict = {}
res_dict['matrix_world'] = [[obj.matrix_world[i][j] for j in range(4)] for i in range(4)]
res_dict['reference_points'] = []
res_dict['groups'] = []
res_dict['weights'] = []

# Get names of all vertex groups
vertex_group_names = {}
print("Vertex groups:")
for i in range(len(obj.vertex_groups)):
    group = obj.vertex_groups[i]
    print(group.index, ":", group.name)
    vertex_group_names[str(group.index)] = group.name
 
for i in range(len(vertices)):
    
    # Add coordinates
    v = obj.matrix_world @ vertices[i].co
    res_dict['reference_points'].append([v[0], v[1], v[2]])
    
    # Add the names of the corresponding vertex groups
    group_list = []
    weight_list = []
    for group in vertices[i].groups:
        group_list.append(vertex_group_names[str(group.group)])
        weight_list.append(group.weight)
    res_dict['groups'].append(group_list)
    res_dict['weights'].append(weight_list)
        
with open(data_file_path, "w") as outfile:
    json.dump(res_dict, outfile)
    print(f'Results saved to "{data_file_path}".')
    