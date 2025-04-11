#!/bin/bash

INSTALL_DIR="$HOME/miniconda3"
CONDA_SCRIPT_PATH="$INSTALL_DIR/etc/profile.d/conda.sh"

echo "Downloading Miniconda installer..."
wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /tmp/miniconda.sh

echo "Installing Miniconda to $INSTALL_DIR..."
bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"
rm -f /tmp/miniconda.sh

echo "Initializing conda for user profile..."
"$INSTALL_DIR/bin/conda" init bash


echo "Sourcing Conda activation script for current session..."
if [ -f "$CONDA_SCRIPT_PATH" ]; then
    source "$CONDA_SCRIPT_PATH"
else
    echo "Error: Conda activation script not found at $CONDA_SCRIPT_PATH"
    exit 1
fi

echo "Verifying the Conda installation..."
conda --version

echo "### Starting environment creation and data download in parallel..."
conda env create -f environment.yml --verbose

echo "### Script finished."
