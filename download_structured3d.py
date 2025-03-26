import os
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

def download_if_needed(url, filepath):
    """Download file if it doesn't exist already"""
    if not filepath.exists():
        print(f"\nDownloading {filepath.name}...")
        try:
            download_file(url, filepath)
        except Exception as e:
            print(f"Error downloading {filepath.name}: {e}")
    else:
        print(f"\n{filepath.name} already exists, skipping download")

def main():
    parser = argparse.ArgumentParser(description='Download Structured3D dataset archives')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of parallel downloads')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path('data/structured3d')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base URL for the dataset files
    base_url = "https://huggingface.co/datasets/Pointcept/structured3d-compressed/resolve/main"
    
    # List of files to download (15 parts)
    files = [f"structured3d_{str(i).zfill(2)}.tar.gz" for i in range(1, 16)]

    print(f"Starting downloads with {args.num_workers} workers...")
    
    # Download files in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for file in files:
            url = f"{base_url}/{file}"
            filepath = output_dir / file
            futures.append(
                executor.submit(download_if_needed, url, filepath)
            )
        
        # Wait for all downloads to complete
        for future in futures:
            future.result()

    print("\nDownload complete!")
    print(f"Archives are available in: {output_dir}")

if __name__ == '__main__':
    main() 