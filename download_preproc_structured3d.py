import os
import tarfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import shutil

def download_and_extract_structured3d():
    """Downloads and extracts Structured3D dataset files from Hugging Face Hub."""
    output_folder = "data/structured3d"
    download_folder = "data/structured3d/downloads"
    repo_id = "Pointcept/structured3d-compressed"
    
    # Create directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(download_folder, exist_ok=True)

    # Download and extract each file
    for i in range(1, 16):
        filename = f"structured3d_{i:02d}.tar.gz"
        local_path = os.path.join(download_folder, filename)

        # Download if file doesn't exist
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=download_folder,
                    local_dir_use_symlinks=False
                )
                if downloaded_path != local_path:
                    shutil.copy2(downloaded_path, local_path)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue

        # Extract the tar file
        print(f"Extracting {filename}...")
        try:
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall(path=output_folder)
        except Exception as e:
            print(f"Error extracting {filename}: {e}")

if __name__ == "__main__":
    download_and_extract_structured3d() 