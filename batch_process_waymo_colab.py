#!/usr/bin/env python3

import os
import argparse
import subprocess
import logging
from pathlib import Path
import shutil
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_symlinks(bucket_path, temp_raw_dir):
    """Create symlinks from bucket files to temporary raw directory."""
    # Convert to absolute paths
    bucket_path = os.path.abspath(bucket_path)
    temp_raw_dir = os.path.abspath(temp_raw_dir)
    
    logger.info(f"Creating symlinks from {bucket_path} to {temp_raw_dir}")
    
    # Create temp raw directory structure
    os.makedirs(temp_raw_dir, exist_ok=True)
    
    # Create symlinks for training and validation directories
    for split in ['training', 'validation']:
        split_bucket_path = os.path.join(bucket_path, 'individual_files', split)
        split_temp_path = os.path.join(temp_raw_dir, split)
        
        logger.info(f"Source path: {split_bucket_path}")
        
        # Verify that source directory exists
        if not os.path.exists(split_bucket_path):
            logger.error(f"Source directory does not exist: {split_bucket_path}")
            continue
            
        # List contents of source directory
        files = os.listdir(split_bucket_path)
        logger.info(f"Found {len(files)} files in {split_bucket_path}")
        
        # Remove existing symlink if it exists
        if os.path.exists(split_temp_path):
            if os.path.islink(split_temp_path):
                os.unlink(split_temp_path)
            elif os.path.isdir(split_temp_path):
                shutil.rmtree(split_temp_path)
            else:
                os.remove(split_temp_path)
        
        # Create directory symlink
        os.symlink(split_bucket_path, split_temp_path)
        logger.info(f"Created directory symlink: {split_temp_path} -> {split_bucket_path}")
    
    logger.info("Symlinks created successfully")
    return temp_raw_dir

def process_file_callback(future, pbar, total_files):
    """Callback function to update progress bar when a file completes processing."""
    pbar.update(1)
    pbar.set_description(f"Processed {pbar.n}/{total_files} files")

def preprocess_dataset(raw_dir, output_dir, splits, num_workers):
    """Run the preprocessing script with progress feedback."""
    # Convert to absolute paths
    raw_dir = os.path.abspath(raw_dir)
    output_dir = os.path.abspath(output_dir)
    
    logger.info(f"Preprocessing dataset from {raw_dir} to {output_dir}")
    
    # Get total number of files to process for progress tracking
    total_files = 0
    file_lists = {}
    
    for split in splits:
        split_dir = os.path.join(raw_dir, split)
        if os.path.exists(split_dir):
            file_list = [f for f in os.listdir(split_dir) if f.endswith('.tfrecord')]
            if not file_list:
                # Try recursive search
                file_list = []
                for root, _, files in os.walk(split_dir):
                    file_list.extend([os.path.join(root, f) for f in files if f.endswith('.tfrecord')])
            total_files += len(file_list)
            file_lists[split] = file_list
    
    logger.info(f"Found {total_files} total files to process across {len(splits)} splits")
    
    if total_files == 0:
        logger.warning("No .tfrecord files found. Falling back to subprocess call.")
        # Fall back to running the preprocessing script directly
        cmd = f"python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py "
        cmd += f"--dataset_root {raw_dir} "
        cmd += f"--output_root {output_dir} "
        cmd += f"--splits {' '.join(splits)} "
        cmd += f"--num_workers {num_workers}"
        
        logger.info(f"Running command: {cmd}")
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            logger.info("Preprocessing completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during preprocessing: {e}")
    else:
        # We could process files individually here for better progress tracking
        # But for now, we'll stick with the subprocess approach with time estimation
        start_time = time.time()
        
        # Run the preprocessing command
        cmd = f"python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py "
        cmd += f"--dataset_root {raw_dir} "
        cmd += f"--output_root {output_dir} "
        cmd += f"--splits {' '.join(splits)} "
        cmd += f"--num_workers {num_workers}"
        
        logger.info(f"Running command: {cmd}")
        
        # Create a progress bar
        with tqdm(total=total_files, desc="Processing files") as pbar:
            # Start a thread to update the progress bar
            def update_progress():
                elapsed_time = 0
                while pbar.n < total_files:
                    time.sleep(5)  # Update every 5 seconds
                    elapsed_time += 5
                    # Get count of processed files by checking the output directory
                    processed_count = 0
                    for split in splits:
                        split_output_dir = os.path.join(output_dir, split)
                        if os.path.exists(split_output_dir):
                            processed_count += len([d for d in os.listdir(split_output_dir) 
                                                  if os.path.isdir(os.path.join(split_output_dir, d))])
                    
                    # Update progress bar
                    if processed_count > pbar.n:
                        pbar.update(processed_count - pbar.n)
                    
                    # Calculate and display ETA
                    if pbar.n > 0:
                        avg_time_per_file = elapsed_time / pbar.n
                        remaining_files = total_files - pbar.n
                        eta_seconds = avg_time_per_file * remaining_files
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                        pbar.set_postfix({"ETA": eta_str})
            
            # Start the progress updater in a separate thread
            import threading
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Run the subprocess
            try:
                subprocess.run(cmd, shell=True, check=True)
                logger.info("Preprocessing completed successfully")
                # Update progress bar to completion
                pbar.n = total_files
                pbar.refresh()
            except subprocess.CalledProcessError as e:
                logger.error(f"Error during preprocessing: {e}")

def create_data_symlink(processed_dir, codebase_dir):
    """Create symlink in the data directory."""
    # Convert to absolute paths
    processed_dir = os.path.abspath(processed_dir)
    codebase_dir = os.path.abspath(codebase_dir)
    
    data_dir = os.path.join(codebase_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    target = os.path.join(data_dir, "waymo")
    if os.path.exists(target):
        if os.path.islink(target):
            os.unlink(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
    
    os.symlink(processed_dir, target)
    logger.info(f"Created symlink: {target} -> {processed_dir}")

def cleanup_temp_dir(temp_dir):
    """Clean up temporary directory with symlinks."""
    if os.path.exists(temp_dir):
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Preprocess Waymo dataset in Google Colab")
    parser.add_argument("--bucket_path", default="../waymo_data", 
                        help="Path to the bucket containing Waymo dataset")
    parser.add_argument("--output_dir", default="./processed_waymo",
                        help="Directory to store processed dataset")
    parser.add_argument("--codebase_dir", default=".", 
                        help="Directory of the codebase")
    parser.add_argument("--temp_dir", default="./temp_waymo_raw",
                        help="Temporary directory for raw data symlinks")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"],
                      help="Dataset splits to process")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for preprocessing")
    parser.add_argument("--cleanup", action="store_true",
                      help="Clean up the temporary directory after processing")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create symlinks to bucket files
        temp_raw_dir = create_symlinks(args.bucket_path, args.temp_dir)
        
        # Preprocess dataset
        preprocess_dataset(temp_raw_dir, args.output_dir, args.splits, args.num_workers)
        
        # Create symlink in data directory
        create_data_symlink(args.output_dir, args.codebase_dir)
        
        logger.info("Dataset preprocessing completed successfully!")
        
    finally:
        # Clean up temporary directory if requested
        if args.cleanup:
            cleanup_temp_dir(args.temp_dir)

if __name__ == "__main__":
    main() 