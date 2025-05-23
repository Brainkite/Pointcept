"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from pathlib import Path
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List
import time


def create_lidar(frame):
    """Parse and save the lidar data in psd format.
    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
    Returns:
        velodyne: point cloud data
        valid_masks: valid masks for each lidar
        points_per_lidar: array of number of points per lidar
    """
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points, valid_masks = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True,
    )
    points_ri2, cp_points_ri2, valid_masks_ri2 = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1,
        keep_polar_features=True,
    )

    # Store number of points per lidar for both first and second returns
    points_per_lidar = np.array([
        len(points[i]) + len(points_ri2[i]) 
        for i in range(len(points))
    ], dtype=np.uint32)

    # 3d points in vehicle frame.
    assert len(points)==len(points_ri2)

    combined_points = [np.concatenate([points[i], points_ri2[i]], axis=0) for i in range(len(points))]
    points_all = np.concatenate(combined_points, axis=0)

    # points_all = np.concatenate(points, axis=0)
    # points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # points_all = np.concatenate([points_all, points_all_ri2], axis=0)

    velodyne = np.c_[points_all[:, 3:6], points_all[:, 1]]
    velodyne = velodyne.reshape((velodyne.shape[0] * velodyne.shape[1]))

    valid_masks = [valid_masks, valid_masks_ri2]
    return velodyne, valid_masks, points_per_lidar


def create_label(frame):
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels
    )
    point_labels_ri2 = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels, ri_index=1
    )

    combined_labels = [np.concatenate([point_labels[i], point_labels_ri2[i]], axis=0) for i in range(len(point_labels))]
    point_labels_all = np.concatenate(combined_labels, axis=0)

    # # point labels.
    # point_labels_all = np.concatenate(point_labels, axis=0)
    # point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    # point_labels_all = np.concatenate([point_labels_all, point_labels_all_ri2], axis=0)

    labels = point_labels_all
    return labels


def convert_range_image_to_cartesian(
    frame, range_images, range_image_top_pose, ri_index=0, keep_polar_features=False
):
    """Convert range images from polar coordinates to Cartesian coordinates.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
        will be 3 if keep_polar_features is False (x, y, z) and 6 if
        keep_polar_features is True (range, intensity, elongation, x, y, z).
    """
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4])
    )

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims,
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    for c in frame.context.laser_calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims
        )
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local,
        )

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        if keep_polar_features:
            # If we want to keep the polar coordinate features of range, intensity,
            # and elongation, concatenate them to be the initial dimensions of the
            # returned Cartesian range image.
            range_image_cartesian = tf.concat(
                [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1
            )

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images


def convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose,
    ri_index=0,
    keep_polar_features=False,
):
    """Convert range images to point cloud.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
      points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        (NOTE: Will be {[N, 6]} if keep_polar_features is true.
      cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    valid_masks = []

    cartesian_range_images = convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features
    )

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(
            range_image_cartesian, tf.compat.v1.where(range_image_mask)
        )

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        valid_masks.append(range_image_mask.numpy())

    return points, cp_points, valid_masks


def convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels, ri_index=0
):
    """Convert segmentation labels from range images to point clouds.

    Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
    range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

    Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
    points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def extract_frame_metadata(frame):
    """Extract metadata from a Waymo frame.
    Args:
        frame: Waymo frame proto
    Returns:
        dict: Dictionary containing frame metadata
    """
    # Get context stats
    stats = frame.context.stats

    # Extract segment_id safely
    segment_id = None
    try:
        if '-' in frame.context.name:
            segment_id = frame.context.name.split('-')[1].split('_')[0]
    except (IndexError, AttributeError):
        pass

    metadata = {
        'timestamp': frame.timestamp_micros,
        'context_name': frame.context.name,
        
        # Location information 
        'location': {
            'segment_id': segment_id,
            'time_of_day': stats.time_of_day if hasattr(stats, 'time_of_day') else None,
            'location': stats.location if hasattr(stats, 'location') else None,
        },
        
        # Environmental conditions
        'conditions': {
            'weather': stats.weather if hasattr(stats, 'weather') else None,
        },
        
        # Special conditions and scene contents
        'scene_contents': {
            'construction': stats.has_construction if hasattr(stats, 'has_construction') else None,
            'pedestrians': stats.has_pedestrians if hasattr(stats, 'has_pedestrians') else None,
            'cyclists': stats.has_cyclists if hasattr(stats, 'has_cyclists') else None,
        },
        
        # Sensor configuration
        'sensors': {
            'num_lidars': len(frame.lasers),
            'num_cameras': len(frame.context.camera_calibrations),
            'lidar_names': [open_dataset.LaserName.Name.Name(laser.name) for laser in frame.lasers],
            'camera_names': [open_dataset.CameraName.Name.Name(cam.name) for cam in frame.context.camera_calibrations],
        },
        
        # Vehicle pose and motion
        'pose': {
            'transform': [float(x) for x in frame.pose.transform],  # 4x4 transformation matrix
        }
    }
    return metadata


def handle_process(file_path: str, output_root: str, test_frame_list: List[str], file_index: int = None, total_files: int = None):
    """Process a single Waymo file."""
    file = os.path.basename(file_path)
    split = os.path.basename(os.path.dirname(file_path))
    
    progress_str = ""
    if file_index is not None and total_files is not None:
        progress_str = f"[{file_index}/{total_files}] ({file_index/total_files*100:.1f}%)"
        print(f"{progress_str} Parsing {split}/{file}")
    else:
        print(f"Parsing {split}/{file}")
    
    save_path = Path(output_root) / split / file.split(".")[0]
    
    metadata_list = []  # Local metadata list for this file
    
    print(f"{progress_str} Loading TFRecord dataset {file}...")
    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    frame_count = 0
    processed_frame_count = 0
    
    # Count total frames first
    print(f"{progress_str} Counting total frames in {file} (may take a while)...")
    start_time = time.time()
    for _ in data_group:
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"{progress_str} Counted {frame_count} frames so far... ({time.time() - start_time:.1f}s)")
    
    print(f"{progress_str} Found {frame_count} total frames in {file} ({time.time() - start_time:.1f}s)")
    
    # Reset dataset iterator
    print(f"{progress_str} Starting to process frames in {file}...")
    data_group = tf.data.TFRecordDataset(file_path, compression_type="")
    
    start_time = time.time()
    for i, data in enumerate(data_group):
        frame_start_time = time.time()
        
        # Display frame processing progress periodically
        if frame_count > 0 and (i % max(1, frame_count // 20) == 0 or i == frame_count - 1):
            elapsed = time.time() - start_time
            frames_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            est_remaining = (frame_count - i - 1) / frames_per_sec if frames_per_sec > 0 else 0
            print(f"{progress_str} Processing file {file}: frame {i+1}/{frame_count} ({(i+1)/frame_count*100:.1f}%) - "
                  f"{frames_per_sec:.1f} frames/s, est. {est_remaining:.1f}s remaining")

        # Parse frame
        parse_start = time.time()
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        context_name = frame.context.name
        timestamp = str(frame.timestamp_micros)
        parse_time = time.time() - parse_start

        process_frame = False
        if split != "testing":
            # for training and validation frame, extract labelled frame
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                process_frame = True
        else:
            # for testing frame, extract frame in test_frame_list
            if f"{context_name},{timestamp}" in test_frame_list:
                process_frame = True
                
        if not process_frame:
            continue
            
        processed_frame_count += 1
        
        # For detailed logging, only log every Nth frame or significant frames
        verbose_log = (processed_frame_count % 10 == 0) or (processed_frame_count <= 3) or (i == frame_count - 1)
        
        if verbose_log:
            print(f"{progress_str} Creating directory for frame {i+1}/{frame_count}, timestamp {timestamp}...")
        os.makedirs(save_path / timestamp, exist_ok=True)

        # extract frame pass above check
        if verbose_log:
            print(f"{progress_str} Extracting lidar data for frame {i+1}/{frame_count}...")
        lidar_start = time.time()
        point_cloud, valid_masks, points_per_lidar = create_lidar(frame)
        point_cloud = point_cloud.reshape(-1, 4)
        coord = point_cloud[:, :3]
        strength = np.tanh(point_cloud[:, -1].reshape([-1, 1]))
        pose = np.array(frame.pose.transform, np.float32).reshape(4, 4)
        mask = np.array(valid_masks, dtype=object)
        lidar_time = time.time() - lidar_start

        # Extract metadata and add frame path
        if verbose_log:
            print(f"{progress_str} Extracting metadata for frame {i+1}/{frame_count}...")
        metadata_start = time.time()
        metadata = extract_frame_metadata(frame)
        metadata['frame_path'] = str(save_path / timestamp)
        metadata_list.append(metadata)
        metadata_time = time.time() - metadata_start

        # Save all extracted data
        if verbose_log:
            print(f"{progress_str} Saving point cloud data for frame {i+1}/{frame_count} ({len(coord)} points)...")
        save_start = time.time()
        np.save(save_path / timestamp / "coord.npy", coord)
        np.save(save_path / timestamp / "strength.npy", strength)
        np.save(save_path / timestamp / "pose.npy", pose)
        np.save(save_path / timestamp / "points_per_lidar.npy", points_per_lidar)

        # save mask for reverse prediction
        if split != "training":
            np.save(save_path / timestamp / "mask.npy", mask)

        # save label
        if split != "testing":
            if verbose_log:
                print(f"{progress_str} Extracting and saving labels for frame {i+1}/{frame_count}...")
            label_start = time.time()
            label = create_label(frame)[:, 1].reshape([-1]) - 1
            np.save(save_path / timestamp / "segment.npy", label)
            label_time = time.time() - label_start
            if verbose_log:
                print(f"{progress_str} Label extraction took {label_time:.2f}s")
        save_time = time.time() - save_start
        
        frame_total_time = time.time() - frame_start_time
        
        # Print detailed timing for significant frames
        if verbose_log:
            print(f"{progress_str} Frame {i+1}/{frame_count} timing: "
                  f"parse={parse_time:.2f}s, lidar={lidar_time:.2f}s, "
                  f"metadata={metadata_time:.2f}s, save={save_time:.2f}s, "
                  f"total={frame_total_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"{progress_str} Completed {split}/{file}: processed {processed_frame_count}/{frame_count} frames in {total_time:.1f}s "
          f"({processed_frame_count/total_time:.1f} frames/s)")
        
    # Return metadata for this file
    return metadata_list


def save_metadata(metadata_list: List[Dict], output_root: str, split: str):
    """Save metadata to Parquet file.
    Args:
        metadata_list: List of metadata dictionaries
        output_root: Root directory for output
        split: Dataset split (training/validation/testing)
    """
    if not metadata_list:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Flatten nested dictionaries
    df = pd.json_normalize(df.to_dict('records'))
    
    # Save to Parquet
    output_path = Path(output_root) / split / "metadata.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved metadata for {split} split to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, help='Path to the Waymo dataset')
    parser.add_argument('--output_root', required=True, help='Path to save the processed dataset')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'], help='Dataset splits to process')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for preprocessing')
    args = parser.parse_args()

    # Display overall settings
    print(f"=== Waymo Dataset Preprocessing ===")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output root: {args.output_root}")
    print(f"Splits: {args.splits}")
    print(f"Number of workers: {args.num_workers}")
    print(f"=====================================")

    start_time = time.time()
    total_files_all_splits = 0
    file_lists_by_split = {}
    
    # First pass: count total files and find all files
    print("Searching for .tfrecord files...")
    for split in args.splits:
        split_start_time = time.time()
        split_dir = os.path.join(args.dataset_root, split)
        print(f"Searching in {split_dir}...")
        file_list = glob.glob(os.path.join(split_dir, '*.tfrecord'))
        
        # If no files found directly, try to find them recursively
        if not file_list:
            print(f"No files found directly in {split_dir}, searching recursively...")
            file_list = glob.glob(os.path.join(split_dir, '**/*.tfrecord'), recursive=True)
        
        if not file_list:
            print(f"No files found in {split_dir}")
            continue
        
        split_search_time = time.time() - split_start_time
        print(f"Found {len(file_list)} files in {split} split (search took {split_search_time:.2f}s)")
        total_files_all_splits += len(file_list)
        file_lists_by_split[split] = file_list

    print(f"Total files to process across all splits: {total_files_all_splits}")
    
    # Load test frame list
    print("Loading test frame list...")
    test_frame_file = os.path.join(
        os.path.dirname(__file__), "3d_semseg_test_set_frames.txt"
    )
    test_frame_list = [x.rstrip() for x in (open(test_frame_file, "r").readlines())]
    print(f"Loaded {len(test_frame_list)} test frames")
    
    processed_files_count = 0
    
    # Process each split
    for split in args.splits:
        if split not in file_lists_by_split:
            continue
            
        file_list = file_lists_by_split[split]
        split_start_time = time.time()
        print(f"\n[Split: {split}] Processing {len(file_list)} files...")

        # Create output directories
        output_dir = os.path.join(args.output_root, split)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Preprocess data
        print(f"Processing {split} scenes with {args.num_workers} workers...")
        
        # Create additional parameters for tracking progress
        file_indices = [processed_files_count + i + 1 for i in range(len(file_list))]
        total_files_list = [total_files_all_splits] * len(file_list)
        
        # Process files using parallel workers and collect metadata
        pool_start_time = time.time()
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            print(f"Starting parallel processing with {args.num_workers} workers...")
            results = list(
                pool.map(
                    handle_process,
                    file_list,
                    repeat(args.output_root),
                    repeat(test_frame_list),
                    file_indices,
                    total_files_list,
                )
            )
        pool_time = time.time() - pool_start_time
        print(f"Parallel processing completed in {pool_time:.2f}s")
        
        # Flatten the list of metadata lists
        all_metadata = []
        print("Gathering metadata from all processed files...")
        for file_metadata in results:
            all_metadata.extend(file_metadata)
        
        # Save metadata for this split
        print(f"Saving metadata for {split} split...")
        save_metadata(all_metadata, args.output_root, split)
        
        processed_files_count += len(file_list)
        split_time = time.time() - split_start_time
        print(f"[Split: {split}] Completed processing {len(file_list)} files in {split_time:.2f}s "
              f"({len(file_list)/split_time:.2f} files/s).")
    
    total_time = time.time() - start_time
    print(f"\n=== Processing Complete ===")
    print(f"Processed {processed_files_count}/{total_files_all_splits} files.")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average processing time per file: {total_time/processed_files_count:.2f}s")
    print(f"Results saved to: {args.output_root}")
    print(f"=============================")


if __name__ == "__main__":
    main()
