import os
import tarfile
from tqdm import tqdm
from datasets import DownloadManager
import concurrent.futures
from typing import List, Tuple

def download_file(url: str, local_path: str) -> Tuple[str, bool]:
    """Downloads a file directly without symbolic links."""
    try:
        download_manager = DownloadManager(local_files_only=False)
        downloaded_path = download_manager.download(url)
        
        # Directly copy the file to our desired location
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            os.remove(local_path)
            
        # Instead of rename (which might create symlinks), read and write the file
        with open(downloaded_path, 'rb') as src, open(local_path, 'wb') as dst:
            dst.write(src.read())
        
        return local_path, True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return local_path, False

def extract_tar(tar_path: str, output_folder: str) -> Tuple[str, bool]:
    """Extracts a tar.gz file into a specified folder with progress."""
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            total_size = sum(member.size for member in members)
            
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Extracting {os.path.basename(tar_path)}") as pbar:
                for member in members:
                    tar.extract(member, path=output_folder)
                    pbar.update(member.size)
                    
        print(f"Successfully extracted {tar_path}")
        return tar_path, True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return tar_path, False

def get_file_list() -> List[Tuple[str, str, str]]:
    """Generate list of files to download with their URLs and local paths."""
    base_url = "https://huggingface.co/datasets/Pointcept/structured3d-compressed/resolve/main"
    download_folder = "data/structured3d/downloads"
    files = []
    
    # Generate file information for files 01-99
    for i in range(1, 100):
        file_name = f"structured3d_{i:02d}.tar.gz"
        url = f"{base_url}/{file_name}"
        local_path = os.path.join(download_folder, file_name)
        files.append((file_name, url, local_path))
    
    return files

def download_and_extract_structured3d(num_threads: int = 1):
    """Downloads and extracts all Structured3D dataset files using multiple threads."""
    output_folder = "data/structured3d"
    download_folder = "data/structured3d/downloads"
    
    # Create directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(download_folder, exist_ok=True)

    files = get_file_list()
    downloaded_files = []

    # Download files using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_url = {}
        for file_name, url, local_path in files:
            if not os.path.exists(local_path):
                future = executor.submit(download_file, url, local_path)
                future_to_url[future] = (file_name, local_path)
            else:
                print(f"File already exists: {local_path}")
                downloaded_files.append(local_path)

        # Process completed downloads
        for future in concurrent.futures.as_completed(future_to_url):
            file_name, local_path = future_to_url[future]
            try:
                path, success = future.result()
                if success:
                    downloaded_files.append(path)
                else:
                    print(f"Failed to download {file_name}")
            except Exception as e:
                print(f"Download failed for {file_name}: {e}")

    # Extract downloaded files using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {}
        for local_path in downloaded_files:
            future = executor.submit(extract_tar, local_path, output_folder)
            future_to_file[future] = local_path

        # Process completed extractions
        for future in concurrent.futures.as_completed(future_to_file):
            local_path = future_to_file[future]
            try:
                path, success = future.result()
                if not success:
                    print(f"Failed to extract {os.path.basename(local_path)}")
            except Exception as e:
                print(f"Extraction failed for {os.path.basename(local_path)}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download and extract Structured3D dataset')
    parser.add_argument('--num_threads', type=int, default=1,
                      help='Number of parallel threads for downloading and extracting (default: 1)')
    args = parser.parse_args()
    
    print(f"Starting download and extraction with {args.num_threads} threads...")
    download_and_extract_structured3d(args.num_threads) 