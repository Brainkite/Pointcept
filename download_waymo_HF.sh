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
    pip install huggingface_hub hf_transfer
fi

# Dataset repository ID
DATASET_ID="Brainkite/waymo_processed"

# Create output directory if it doesn't exist
OUTPUT_DIR="waymo_dataset"
mkdir -p "$OUTPUT_DIR"

echo "Downloading Waymo dataset from $DATASET_ID..."
echo "Using hf_transfer for faster downloads..."
echo "This may take a while depending on your internet connection..."

# Maximum number of retry attempts
MAX_RETRIES=5
# Initial retry count
RETRY_COUNT=0
# Retry delay in seconds
RETRY_DELAY=3

# Download with retry loop
download_successful=false
while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$download_successful" = false ]; do
    if [ $RETRY_COUNT -gt 0 ]; then
        echo "Retry attempt $RETRY_COUNT of $MAX_RETRIES after waiting for $RETRY_DELAY seconds..."
    fi
    
    huggingface-cli download "$DATASET_ID" \
        --repo-type dataset \
        --token "$HUGGINGFACE_TOKEN" \
        --local-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Dataset downloaded successfully to $OUTPUT_DIR"
        download_successful=true
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "Download failed. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            # Increase retry delay for subsequent attempts
            RETRY_DELAY=$((RETRY_DELAY + 3))
        else
            echo "Error: Failed to download the dataset after $MAX_RETRIES attempts"
            exit 1
        fi
    fi
done 