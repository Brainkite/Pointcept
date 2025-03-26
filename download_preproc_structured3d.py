import os
import tarfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import concurrent.futures
import shutil

def download_file(filename: str, repo_id: str, download_folder: str) -> bool:
    """Downloads a single file from Hugging Face Hub."""
    local_path = os.path.join(download_folder, filename)
    if os.path.exists(local_path):
        print(f"File already exists: {filename}")
        return local_path
    
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
        return local_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def extract_tar(tar_path: str, output_folder: str) -> bool:
    """Extracts a tar.gz file with progress bar."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc=f"Extracting {os.path.basename(tar_path)}") as pbar:
                for member in members:
                    tar.extract(member, path=output_folder)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def download_and_extract_structured3d(num_workers: int = 4):
    """Downloads and extracts Structured3D dataset files in parallel."""
    output_folder = "data/structured3d"
    download_folder = "data/structured3d/downloads"
    repo_id = "Pointcept/structured3d-compressed"
    
    # Create directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(download_folder, exist_ok=True)

    # Prepare file list
    filenames = [f"structured3d_{i:02d}.tar.gz" for i in range(1, 16)]
    
    # Download files in parallel
    downloaded_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(download_file, filename, repo_id, download_folder): filename 
            for filename in filenames
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            local_path = future.result()
            if local_path:
                downloaded_files.append(local_path)

    # Extract files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(lambda f: extract_tar(f, output_folder), downloaded_files)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, 
                      help='Number of parallel workers (default: 4)')
    args = parser.parse_args()
    
    download_and_extract_structured3d(args.workers) 