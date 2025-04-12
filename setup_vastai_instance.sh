#!/bin/bash

set -e

cd /workspace/Pointcept

echo "### DOWNLOAD WAYMO DATASET"
bash ./download_waymo_HF.sh

echo "### EXTRACT WAYMO DATASET"
bash extract_processed_waymo.sh

rm -rf waymo_dataset

echo "### INSTALL ENV AND TRAIN"
bash ./setup_pointcept_env.sh && bash ./train_ptv3_waymo.sh
