#!/bin/bash

# Check if HUGGINGFACE_TOKEN is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is not set"
    echo "Please set your Hugging Face token using:"
    echo "export HUGGINGFACE_TOKEN='your_token_here'"
    exit 1
fi

# Enable hf_transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check if huggingface_hub is installed with hf_transfer
if ! pip show huggingface_hub | grep -q "hf_transfer"; then
    echo "Installing huggingface_hub with hf_transfer support..."
    pip install "huggingface_hub[hf_transfer]"
fi

# Dataset repository ID
DATASET_ID="Brainkite/waymo_processed"

# Create output directory if it doesn't exist
OUTPUT_DIR="waymo_dataset"
mkdir -p "$OUTPUT_DIR"

echo "Downloading Waymo dataset from $DATASET_ID..."
echo "Using hf_transfer for faster downloads..."
echo "This may take a while depending on your internet connection..."

# Download the dataset using huggingface-cli with hf_transfer
# Specify repo_type as dataset since this is a dataset repository
huggingface-cli download "$DATASET_ID" \
    --repo-type dataset \
    --token "$HUGGINGFACE_TOKEN" \
    --local-dir "$OUTPUT_DIR"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Dataset downloaded successfully to $OUTPUT_DIR"
else
    echo "Error: Failed to download the dataset"
    exit 1
fi 