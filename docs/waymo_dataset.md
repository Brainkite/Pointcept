# Waymo Dataset Documentation

This document describes the structure and usage of the processed Waymo dataset, including how to work with multi-sensor point clouds and metadata.

## Dataset Overview

The Waymo Open Dataset is a large-scale dataset for autonomous driving research. Our processed version provides:

1. **Point cloud data**: Pre-processed from 5 lidar sensors
2. **Segmentation labels**: For semantic segmentation tasks 
3. **Pose information**: Vehicle poses for each frame
4. **Rich metadata**: Weather, location, time of day, and other contextual information

## Point Cloud Organization

### Multi-Sensor Point Cloud Structure

The Waymo dataset uses 5 different lidar sensors:
1. TOP (mid-range lidar)
2. FRONT_LEFT
3. FRONT_RIGHT
4. SIDE_LEFT
5. SIDE_RIGHT

Each sensor provides points in both first and second returns. The points from all sensors are concatenated into a single array, but we can track which points came from which sensor using the `points_per_lidar.npy` file.

### Using points_per_lidar.npy

The `points_per_lidar.npy` file contains an array of 5 integers, where each integer represents the total number of points (first + second returns) from each lidar sensor in order:
```python
points_per_lidar = [top_points, front_left_points, front_right_points, side_left_points, side_right_points]
```

#### Example: Filtering Points from a Single Sensor

```python
import numpy as np

# Load point cloud data
coord = np.load('coord.npy')  # shape: (N, 3)
strength = np.load('strength.npy')  # shape: (N, 1)
points_per_lidar = np.load('points_per_lidar.npy')  # shape: (5,)

# Calculate cumulative sum to get start indices for each sensor
sensor_start_indices = np.cumsum([0] + points_per_lidar.tolist())

# Get points from TOP lidar (index 0)
top_start = sensor_start_indices[0]
top_end = sensor_start_indices[1]
top_points = coord[top_start:top_end]
top_strength = strength[top_start:top_end]

# Get points from FRONT_LEFT lidar (index 1)
front_left_start = sensor_start_indices[1]
front_left_end = sensor_start_indices[2]
front_left_points = coord[front_left_start:front_left_end]
front_left_strength = strength[front_left_start:front_left_end]
```

## Metadata Structure

The dataset metadata is stored in Parquet format, one file per split (training/validation/testing). This format allows for efficient querying and filtering of the dataset based on various attributes.

### Metadata Fields

Each frame's metadata contains the following information:

#### Basic Information
- `timestamp`: Microsecond timestamp for the frame
- `context_name`: Unique identifier for the segment
- `frame_path`: Path to the frame's data directory
- `segment_id`: ID extracted from the context name (when available)

#### Location Information
- `location.time_of_day`: Time of day
  - Values: "Day", "Dawn/Dusk", "Night"
- `location.location`: Geographic location identifier
  - Examples: "location_phx" (Phoenix), "location_sf" (San Francisco), etc.

#### Environmental Conditions
- `conditions.weather`: Weather conditions
  - Values include: "sunny", "cloudy", "rainy", "foggy", etc.

#### Scene Contents
- `scene_contents.construction`: Indicates presence of construction (boolean)
- `scene_contents.pedestrians`: Indicates presence of pedestrians (boolean)
- `scene_contents.cyclists`: Indicates presence of cyclists (boolean)

#### Sensor Configuration
- `sensors.num_lidars`: Number of active lidar sensors (typically 5)
- `sensors.num_cameras`: Number of active cameras (typically 5)
- `sensors.lidar_names`: Names of the active lidar sensors
  - Example: ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]
- `sensors.camera_names`: Names of the active cameras
  - Example: ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

#### Vehicle Pose
- `pose.transform`: 4x4 transformation matrix (flattened to 16 values)

### Example: Filtering Frames by Conditions

```python
import pandas as pd

# Load metadata for a specific split
df = pd.read_parquet('data/waymo_processed/validation/metadata.parquet')

# Filter frames with sunny weather during day time
sunny_day_frames = df[
    (df['conditions.weather'] == 'sunny') & 
    (df['location.time_of_day'] == 'Day')
]

# Get the frame paths
frame_paths = sunny_day_frames['frame_path'].tolist()

# Example: Load and process filtered frames
for frame_path in frame_paths:
    # Load point cloud data
    coord = np.load(f"{frame_path}/coord.npy")
    strength = np.load(f"{frame_path}/strength.npy")
    points_per_lidar = np.load(f"{frame_path}/points_per_lidar.npy")
    
    # Process the frame...
```

### Additional Filtering Examples

```python
# Get all frames from Phoenix
phoenix_frames = df[df['location.location'] == 'location_phx']

# Get frames with cyclists
cyclist_frames = df[df['scene_contents.cyclists'] == True]

# Get frames with all 5 lidars active
full_sensor_frames = df[df['sensors.num_lidars'] == 5]

# Get frames with specific weather conditions
bad_weather_frames = df[
    ~df['conditions.weather'].isin(['sunny', 'clear'])
]
```

## Dataset Processing

The dataset is processed using two main scripts:

1. `download_waymo.py`: Downloads the raw dataset files from the Waymo Open Dataset
2. `batch_process_waymo.py`: Processes the raw data into a structured format

### Example: Processing the Waymo Dataset

```bash
# Download the dataset (with subset size of 2 and 10 parallel workers)
./download_waymo.py \
    --raw_dir data/waymo_raw \
    --num_download_workers 10 \
    --subset_size 2 \
    --cleanup

# Process the dataset (with 50 parallel workers)
./batch_process_waymo.py \
    --raw_dir data/waymo_raw \
    --output_dir data/waymo_processed \
    --codebase_dir . \
    --num_workers 50 \
    --cleanup
```

## Recent Waymo Dataset Updates

The Waymo Open Dataset has received several updates since its initial release:

### March 2024 Update (v1.4.3 and v2.0.1)
- Improved 3D semantic segmentation ground truth labels, especially for motorcyclists
- Added camera data including various views (front, sides, rear)
- Fixed alignment between LiDAR data and roadgraph inputs

### March 2023 Update (v2.0.0)
- Introduced modular format for selective component downloads
- Added 3D map data as polylines or polygons
- Added mask to indicate camera coverage for each pixel

### March 2022 Update (v1.3.0)
- Added 3D semantic segmentation labels
- Added 2D and 3D keypoint labels and metrics
- Added correspondence between 2D (camera) and 3D (lidar) labels

## Notes

1. The point cloud data is organized to maintain the order of points from each sensor, making it easy to filter points by sensor using the `points_per_lidar.npy` file.

2. The metadata is stored in Parquet format for efficient querying and filtering. The nested structure is flattened when saved to Parquet, with dot notation used for nested fields (e.g., `conditions.weather`).

3. When working with the dataset, you can combine both point cloud filtering and metadata filtering to create specific subsets of the data for analysis or training.

4. To work with the latest Waymo Open Dataset features, you may need to update your processing scripts to handle new label types and data structures. 