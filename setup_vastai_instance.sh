#!/bin/bash

set -e

cd /workspace/Pointcept

echo "### DOWNLOAD WAYMO DATASET"
bash ./download_waymo_HF.sh

echo "### EXTRACT WAYMO DATASET"
bash extract_processed_waymo.sh

echo "### INSTALL ENV"
bash ./setup_pointcept_env.sh

echo "### SETUP FINISHED"

echo "### TRAIN"
bash train_ptv3_waymo.sh