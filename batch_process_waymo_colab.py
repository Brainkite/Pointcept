#!/usr/bin/env python3

import os
import argparse
import subprocess
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_symlinks(bucket_path, temp_raw_dir):
    """Create symlinks from bucket files to temporary raw directory."""
    logger.info(f"Creating symlinks from {bucket_path} to {temp_raw_dir}")
    
    # Create temp raw directory structure
    os.makedirs(temp_raw_dir, exist_ok=True)
    
    # Create symlinks for training and validation directories
    for split in ['training', 'validation']:
        split_bucket_path = os.path.join(bucket_path, 'individual_files', split)
        split_temp_path = os.path.join(temp_raw_dir, split)
        
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

def preprocess_dataset(raw_dir, output_dir, splits, num_workers):
    """Run the preprocessing script."""
    cmd = f"python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py "
    cmd += f"--dataset_root {raw_dir} "
    cmd += f"--output_root {output_dir} "
    cmd += f"--splits {' '.join(splits)} "
    cmd += f"--num_workers {num_workers}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info("Preprocessing completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during preprocessing: {e}")

def create_data_symlink(processed_dir, codebase_dir):
    """Create symlink in the data directory."""
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
    parser.add_argument("--bucket_path", default="./waymo_data", 
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