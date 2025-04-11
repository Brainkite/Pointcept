#!/bin/bash

# Script to extract Waymo processed data to the target location

# Define target directory
TARGET_DIR="/workspace/Pointcept/data/waymo"

# Create target directories if they don't exist
mkdir -p "$TARGET_DIR/training"
mkdir -p "$TARGET_DIR/validation"

echo "Starting extraction to $TARGET_DIR..."

# Extract training data
echo "Extracting training data..."
cd waymo_dataset/training
cat processed_waymo_training.tar.zst.part* | \
  tar --use-compress-program=unzstd -T0 -xf - \
      --strip-components=4 \
      -C "$TARGET_DIR/training" \
      --wildcards "content/Pointcept/processed_waymo/training/*"

# Check if training extraction was successful
if [ $? -eq 0 ]; then
    echo "Training data extraction completed successfully."
else
    echo "Error: Training data extraction failed."
    exit 1
fi

# Extract validation data
echo "Extracting validation data..."
cd ../validation
cat processed_waymo_validation.tar.zst | \
  tar --use-compress-program=unzstd -T0 -xf - \
      --strip-components=4 \
      -C "$TARGET_DIR/validation" \
      --wildcards "content/Pointcept/processed_waymo/validation/*"

# Check if validation extraction was successful
if [ $? -eq 0 ]; then
    echo "Validation data extraction completed successfully."
else
    echo "Error: Validation data extraction failed."
    exit 1
fi

# Return to the original directory
cd ..

# Print some stats about the extracted data
echo "Statistics of extracted data:"
echo "Training data:"
find "$TARGET_DIR/training" -type f | wc -l | xargs echo "  Total files:"
du -sh "$TARGET_DIR/training" | awk '{print "  Total size: " $1}'

echo "Validation data:"
find "$TARGET_DIR/validation" -type f | wc -l | xargs echo "  Total files:"
du -sh "$TARGET_DIR/validation" | awk '{print "  Total size: " $1}'

echo "Extraction completed successfully!"
echo "Data extracted to:"
echo "  - $TARGET_DIR/training"
echo "  - $TARGET_DIR/validation"