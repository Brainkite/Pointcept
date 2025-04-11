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

echo "Waymo env Setup completed!" 