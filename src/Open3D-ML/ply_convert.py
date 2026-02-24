from plyfile import PlyData, PlyElement
import numpy as np

# Read the PLY file
ply_data = PlyData.read('src/Open3D-ML/data/Paris_Lille3D/training_50_classes/Lille1.ply')

# Extract vertex data (assuming 'vertex' element)
vertices = ply_data['vertex'].data

# Define new data types (convert uint to int)
new_dtype = []
for prop in ply_data['vertex'].data.dtype.descr:
    if prop[1] == '<u4':  # Check for uint32 (common for PLY_UINT)
        new_dtype.append((prop[0], '<i4'))  # Change to int32
    else:
        new_dtype.append(prop)

# Convert the data to the new dtype
new_vertices = np.array(vertices, dtype=new_dtype)

# Create a new PlyElement
new_vertex_element = PlyElement.describe(new_vertices, 'vertex')

# Replace the original vertex element
new_elements = [new_vertex_element] + [
    elem for elem in ply_data.elements if elem.name != 'vertex'
]

# Save the modified PLY file
PlyData(new_elements, text=ply_data.text).write('src/Open3D-ML/data/Paris_Lille3D/training_50_classes/Lille1_output.ply')