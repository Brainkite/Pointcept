import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm

def download_file(url, filepath):
    """Download a file from a URL to a local filepath with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def extract_tar(filepath):
    """Extract a tar.gz file"""
    subprocess.run(['tar', '-xzf', filepath], check=True)

def main():
    parser = argparse.ArgumentParser(description='Download preprocessed Structured3D dataset')
    parser.add_argument('--output_dir', type=str, default='data/structured3d',
                      help='Output directory for the dataset')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base URL for the dataset files
    base_url = "https://huggingface.co/datasets/Pointcept/structured3d-compressed/resolve/main"
    
    # List of files to download (15 parts)
    files = [f"structured3d_{str(i).zfill(2)}.tar.gz" for i in range(1, 16)]

    print("Starting downloads...")
    
    # Download all files
    for file in files:
        url = f"{base_url}/{file}"
        filepath = output_dir / file
        
        if not filepath.exists():
            print(f"\nDownloading {file}...")
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                continue
        else:
            print(f"\n{file} already exists, skipping download")

    print("\nExtracting files...")
    
    # Extract all files in parallel using a thread pool
    with ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            filepath = output_dir / file
            if filepath.exists():
                futures.append(executor.submit(extract_tar, filepath))
        
        # Wait for all extractions to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error during extraction: {e}")

    print("\nCleaning up downloaded archives...")
    
    # Remove the downloaded tar.gz files
    for file in files:
        filepath = output_dir / file
        if filepath.exists():
            filepath.unlink()

    print("\nDownload and extraction complete!")
    print(f"Dataset is available in: {output_dir}")

if __name__ == '__main__':
    main() 