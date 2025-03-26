import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

def download_file(file):
    """Download a file from Hugging Face Hub"""
    try:
        output_dir = Path('data/structured3d')
        filepath = output_dir / file
        
        if filepath.exists() and filepath.stat().st_size > 1_000_000_000:
            print(f"\n{file} already exists and has valid size, skipping download")
            return filepath
            
        print(f"\nStarting download of {file}...")
        # Use snapshot_download to handle authentication automatically
        snapshot_download(
            repo_id="Pointcept/structured3d-compressed",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=file
        )
        return filepath
    except Exception as e:
        print(f"Error downloading {file}: {e}")
        if filepath.exists():
            filepath.unlink()
        return None

def extract_tar(filepath):
    """Extract a tar.gz file"""
    print(f"\nExtracting {filepath}...")
    try:
        # First verify the file exists and has reasonable size
        if not filepath.exists():
            raise Exception(f"File {filepath} does not exist")
        
        file_size = filepath.stat().st_size
        if file_size < 1_000_000_000:  # 1GB
            raise Exception(f"File {filepath} is too small ({file_size} bytes)")
            
        subprocess.run(['tar', '-xf', filepath], check=True)
        return True
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download preprocessed Structured3D dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of parallel downloads (default: 4)')
    args = parser.parse_args()

    # Set fixed output directory
    output_dir = Path('data/structured3d')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of files to download (15 parts)
    files = [f"structured3d_{str(i).zfill(2)}.tar.gz" for i in range(1, 16)]

    print("Starting downloads...")
    print(f"Files will be downloaded to: {output_dir}")
    print("Using Hugging Face credentials from huggingface-cli login")
    print(f"Using {args.num_workers} parallel downloads")
    
    # Download files in parallel
    downloaded_files = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all download tasks
        future_to_file = {executor.submit(download_file, file): file for file in files}
        
        # Process completed downloads
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                filepath = future.result()
                if filepath:
                    downloaded_files.append(filepath)
            except Exception as e:
                print(f"Download failed for {file}: {e}")

    if not downloaded_files:
        print("\nNo files were successfully downloaded. Please check if you're logged in (run 'huggingface-cli login')")
        return

    print(f"\nSuccessfully downloaded {len(downloaded_files)} files")
    print("\nExtracting files...")
    
    # Extract files sequentially to avoid disk space issues
    successful_extractions = []
    for filepath in downloaded_files:
        if extract_tar(filepath):
            successful_extractions.append(filepath)

    print("\nCleaning up downloaded archives...")
    
    # Remove the successfully extracted archives
    for filepath in successful_extractions:
        try:
            filepath.unlink()
            print(f"Removed {filepath}")
        except Exception as e:
            print(f"Error removing {filepath}: {e}")

    print("\nDownload and extraction complete!")
    print(f"Dataset is available in: {output_dir}")

if __name__ == '__main__':
    main() 