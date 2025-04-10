#!/usr/bin/env python3

import os
import argparse
import subprocess
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_directory(base_dir):
    """Clean up the processing directory if it exists."""
    if os.path.exists(base_dir):
        logger.info(f"Cleaning up existing directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

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

def create_symlink(processed_dir, codebase_dir):
    """Create symlink to the processed dataset."""
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

def main():
    parser = argparse.ArgumentParser(description="Preprocess Waymo dataset")
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to store processed dataset")
    parser.add_argument("--codebase_dir", required=True, help="Directory of the codebase")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"],
                      help="Dataset splits to process")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for preprocessing")
    parser.add_argument("--cleanup", action="store_true",
                      help="Clean up the processing directory before starting")
    
    args = parser.parse_args()
    
    # Clean up if requested
    if args.cleanup:
        cleanup_directory(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess dataset
    preprocess_dataset(args.raw_dir, args.output_dir, args.splits, args.num_workers)
    
    # Create symlink
    create_symlink(args.output_dir, args.codebase_dir)
    
    logger.info("Dataset preprocessing completed successfully!")

if __name__ == "__main__":
    main() 