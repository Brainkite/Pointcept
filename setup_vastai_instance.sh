#!/bin/bash

set -e

cd /workspace

echo "### CLONING POINTCEPT REPO"
git clone https://github.com/Brainkite/Pointcept.git
cd Pointcept

echo "### DOWNLOAD WAYMO DATASET"
bash ./download_waymo_HF.sh

echo "### EXTRACT WAYMO DATASET"
bash ./extract_waymo_dataset.sh

echo "### INSTALL ENV"
bash ./setup_pointcept_env.sh

echo "### SETUP FINISHED"