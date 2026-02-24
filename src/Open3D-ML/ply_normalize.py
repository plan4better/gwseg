import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

GROUP_MAPPING = {
    # Explicit surface mappings
    202030000: 1,   # sidewalk
    202020000: 2,   # road
    
    202060000: 3,   # vegetation,
    304020000: 3,   # tree
    304030000: 3,   # bush
    304040000: 3,   # potted-plant

    202050000: 4,   # island
    203000000: 5,   # building
    302020600: 6,   # traffic-sign
    302020500: 7,   # traffic-light
    
    # Special cases
    0: 0,           # unclassified
    100000000: 0,   # other
}

def process_ply_file(input_path, output_path):
    """Process a single PLY file with class remapping"""
    try:
        # Read input file
        ply = PlyData.read(input_path)
        vertices = ply['vertex'].data
        
        # Create new dtype with original_class and class
        new_dtype = []
        for descr in vertices.dtype.descr:
            if descr[0] == 'class':
                new_dtype.append(('original_class', descr[1]))  # Preserve original
            else:
                new_dtype.append(descr)
        new_dtype.append(('class', '<i4'))  # Add new class
        
        # Create new vertex array
        new_vertices = np.zeros(vertices.shape, dtype=new_dtype)
        
        # Copy data and remap classes
        for name in vertices.dtype.names:
            if name == 'class':
                new_vertices['original_class'] = vertices[name]
            else:
                new_vertices[name] = vertices[name]
        
        # Vectorized class remapping (default to 0 for unmapped classes)
        remap_fn = np.vectorize(lambda x: GROUP_MAPPING.get(x, 0), otypes=[np.uint16])
        new_vertices['class'] = remap_fn(new_vertices['original_class'])
        
        # Preserve all elements
        new_elements = [PlyElement.describe(new_vertices, 'vertex')]
        for element in ply.elements:
            if element.name != 'vertex':
                new_elements.append(element)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData(new_elements, text=ply.text).write(output_path)
        print(f"Processed: {input_path} -> {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_folder(input_dir, output_dir, exclude_files=None):
    """Process all PLY files in a folder"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    exclude_set = set(exclude_files or [])
    
    for ply_file in input_path.glob('*.ply'):
        if ply_file.name in exclude_set:
            print(f"Skipping excluded file: {ply_file.name}")
            continue
            
        output_file = output_path / ply_file.name
        process_ply_file(ply_file, output_file)


if __name__ == "__main__":
    input_dir = 'src/Open3D-ML/data/Paris_Lille3D/training_50_classes_raw'
    output_dir = 'src/Open3D-ML/data/Paris_Lille3D/training_50_classes'
    exclude_files = {}
    

    assert 0 in GROUP_MAPPING, "Missing unlabeled mapping"
    
    # Process files
    process_folder(
        input_dir,
        output_dir,
        exclude_files=exclude_files
    )