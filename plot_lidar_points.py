import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random

# Path to validation dataset
validation_path = "data/waymo/validation/validation"
output_dir = "lidar_plots"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all segments
segments = glob.glob(os.path.join(validation_path, "segment-*"))
segments = segments[:2]
for segment in tqdm(segments, desc="Processing segments"):
    segment_name = os.path.basename(segment)
    segment_output_dir = os.path.join(output_dir, segment_name)
    os.makedirs(segment_output_dir, exist_ok=True)
    
    # Find all frames in segment
    frames = glob.glob(os.path.join(segment, "*"))
    # Limit to first 10 frames
    for frame in tqdm(frames, desc=f"Processing frames in {segment_name}", leave=False):
        frame_name = os.path.basename(frame)
        
        # Check if required files exist
        coord_path = os.path.join(frame, "coord.npy")
        points_per_lidar_path = os.path.join(frame, "points_per_lidar.npy")
        
        if not os.path.exists(coord_path) or not os.path.exists(points_per_lidar_path):
            continue
        
        # Load data
        coords = np.load(coord_path)
        points_per_lidar = np.load(points_per_lidar_path)
        
        # Create lidar index array
        lidar_indices = np.zeros(coords.shape[0], dtype=int)
        
        start_idx = 0
        for lidar_idx, point_count in enumerate(points_per_lidar):
            end_idx = start_idx + point_count
            lidar_indices[start_idx:end_idx] = lidar_idx
            start_idx = end_idx
        
        # Create mask for points from lidars 1-4 (skip the first lidar)
        mask = lidar_indices > 0
        filtered_coords = coords[mask]
        filtered_lidar_indices = lidar_indices[mask]
        
        # Sample 2000 random points (or all if less than 2000)
        num_points = min(2000, filtered_coords.shape[0])
        if filtered_coords.shape[0] > num_points:
            random_indices = random.sample(range(filtered_coords.shape[0]), num_points)
            sampled_coords = filtered_coords[random_indices]
            sampled_lidar_indices = filtered_lidar_indices[random_indices]
        else:
            sampled_coords = filtered_coords
            sampled_lidar_indices = filtered_lidar_indices
        
        # Create plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            sampled_coords[:, 0],
            sampled_coords[:, 1],
            c=sampled_lidar_indices,
            cmap='tab10',
            s=1,
            alpha=0.7
        )
        
        plt.title(f"LiDAR Point Cloud (Excluding LiDAR 0) - {frame_name}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.colorbar(scatter, label='LiDAR ID')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = os.path.join(segment_output_dir, f"{frame_name}.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

print(f"Plots saved to {output_dir}") 