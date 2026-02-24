import os
import numpy as np
from plyfile import PlyData
from ml3d.datasets.parislille3d_50_classes import label_to_names


def count_points_per_class(folder_path, label_to_names, exclude_files=None):
    """
    Count total points per class across all PLY files in a folder.
    
    Args:
        folder_path (str): Path to folder containing PLY files
        label_to_names (dict): Dictionary mapping class IDs to names
        exclude_files (set/list): Collection of filenames to exclude (e.g., {'file1.ply', 'file2.ply'})
        
    Returns:
        np.ndarray: Array of counts in the order of label_to_names.values()
    """
    # Get ordered class IDs and create index mapping
    class_ids = list(label_to_names.keys())
    num_classes = len(class_ids)
    counts = np.zeros(num_classes, dtype=np.int64)
    label_to_index = {label: idx for idx, label in enumerate(class_ids)}
    # Convert exclude_files to set for fast lookups
    exclude_files = set(exclude_files) if exclude_files else set()

    # Process each PLY file
    for filename in os.listdir(folder_path):
        if filename.endswith('.ply') and filename not in exclude_files:
            file_path = os.path.join(folder_path, filename)
            try:
                # Read PLY file
                ply_data = PlyData.read(file_path)
                vertices = ply_data['vertex']
                
                # Extract class (assuming 'class' property exists)
                classes = vertices['class'].astype(int)
                
                # Count occurrences using numpy for efficiency
                unique_classes, counts_per_class = np.unique(classes, return_counts=True)
                
                # Accumulate counts
                for label, count in zip(unique_classes, counts_per_class):
                    if label in label_to_index:
                        counts[label_to_index[label]] += count
                        
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                
    return counts

counts = count_points_per_class('src/Open3D-ML/data/Paris_Lille3D/training_50_classes', label_to_names, exclude_files={'Lille2.ply'})
print(counts)