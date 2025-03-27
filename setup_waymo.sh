#!/bin/bash

# Create and activate conda environment
conda create -n waymo python=3.10 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo

# Install Google Cloud SDK (if not already installed)
if ! command -v gcloud &> /dev/null; then
    echo "Installing Google Cloud SDK..."
    # Add the Cloud SDK distribution URI as a package source
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

    # Import the Google Cloud public key
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

    # Update and install the Cloud SDK
    sudo apt-get update && sudo apt-get install google-cloud-sdk
fi

# Initialize gcloud (will prompt for login)
gcloud init

# Install waymo-open-dataset package
pip install waymo-open-dataset-tf-2-12-0

# Install additional required packages
pip install numpy tensorflow tensorflow-gpu tqdm

# Create directories for the dataset
mkdir -p data/waymo_raw
mkdir -p data/waymo_processed

# Step 1: Download the raw data with subset size of 2
echo "Starting download with subset size of 2..."
./download_waymo.py \
    --raw_dir data/waymo_raw \
    --num_download_workers 10 \
    --subset_size 2 \
    --cleanup

# Step 2: Process the raw data
echo "Starting processing of downloaded data..."
./batch_process_waymo.py \
    --raw_dir data/waymo_raw \
    --output_dir data/waymo_processed \
    --codebase_dir . \
    --num_workers 50 \
    --cleanup

echo "Setup, download, and processing completed!" 