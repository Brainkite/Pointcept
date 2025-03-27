#!/bin/bash

# Create directories for the dataset
mkdir -p data/waymo_raw
mkdir -p data/waymo_processed

# Step 1: Download the raw data with subset size of 2
echo "Starting download"
./download_waymo.py \
    --raw_dir data/waymo_raw \
    --num_download_workers 10 \
    # --subset_size 0 \
    --cleanup

# Step 2: Process the raw data
echo "Starting processing of downloaded data..."
./batch_process_waymo.py \
    --raw_dir data/waymo_raw \
    --output_dir data/waymo_processed \
    --codebase_dir . \
    --num_workers 50 \
    --cleanup

echo "Download, and processing completed!" 