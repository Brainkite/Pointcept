#!/usr/bin/env python3

import os
import argparse
import subprocess
import random
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import concurrent.futures
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_directory(base_dir):
    """Clean up the download directory if it exists."""
    if os.path.exists(base_dir):
        logger.info(f"Cleaning up existing directory: {base_dir}")
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

def setup_directories(base_dir):
    """Create necessary directories for the dataset."""
    dirs = ['training', 'validation']
    for dir_name in dirs:
        Path(os.path.join(base_dir, dir_name)).mkdir(parents=True, exist_ok=True)

def download_file(file, split_dir):
    """Download a single file using gsutil."""
    try:
        subprocess.run(f"gsutil cp {file} {split_dir}/", shell=True, check=True)
        return True, file
    except subprocess.CalledProcessError as e:
        return False, f"Error downloading {file}: {e}"

def download_dataset(base_dir, splits=['training', 'validation'], subset_size=None, num_download_workers=4):
    """Download Waymo dataset files using gsutil with parallel downloads."""
    bucket = "gs://waymo_open_dataset_v_1_4_3/individual_files"
    
    for split in splits:
        logger.info(f"Downloading {split} split...")
        split_dir = os.path.join(base_dir, split)
        
        # Get list of files in the split
        cmd = f"gsutil ls {bucket}/{split}/"
        try:
            files = subprocess.check_output(cmd, shell=True).decode().splitlines()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error listing files: {e}")
            continue

        # If subset_size is specified, randomly select files
        if subset_size is not None:
            if subset_size > len(files):
                logger.warning(f"Subset size {subset_size} is larger than total files ({len(files)}) for {split}. Using all files.")
            else:
                files = random.sample(files, subset_size)
                logger.info(f"Randomly selected {subset_size} files for {split} split")

        # Create a partial function with the split_dir argument
        download_func = partial(download_file, split_dir=split_dir)
        
        # Download files in parallel with progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_download_workers) as executor:
            futures = [executor.submit(download_func, file) for file in files]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(files), 
                             desc=f"Downloading {split} files"):
                success, result = future.result()
                if not success:
                    logger.error(result)

def main():
    parser = argparse.ArgumentParser(description="Download Waymo dataset")
    parser.add_argument("--raw_dir", required=True, help="Directory to store raw dataset")
    parser.add_argument("--splits", nargs="+", default=["training", "validation"],
                      help="Dataset splits to download")
    parser.add_argument("--num_download_workers", type=int, default=4,
                      help="Number of workers for parallel downloads")
    parser.add_argument("--subset_size", type=int, default=None,
                      help="Number of files to randomly select from each split")
    parser.add_argument("--cleanup", action="store_true",
                      help="Clean up the download directory before starting")
    
    args = parser.parse_args()
    
    # Clean up if requested
    if args.cleanup:
        cleanup_directory(args.raw_dir)
    
    # Create directories
    setup_directories(args.raw_dir)
    
    # Download dataset
    download_dataset(args.raw_dir, args.splits, args.subset_size, args.num_download_workers)
    
    logger.info("Dataset download completed successfully!")

if __name__ == "__main__":
    main() 