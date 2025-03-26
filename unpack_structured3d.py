import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def extract_tar(filepath):
    """Extract a tar.gz file"""
    subprocess.run(['tar', '-xzf', filepath], check=True)

def main():
    parser = argparse.ArgumentParser(description='Unpack Structured3D dataset archives')
    parser.add_argument('--keep_archives', action='store_true',
                      help='Keep the archive files after extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of parallel extractions')
    args = parser.parse_args()

    input_dir = Path('data/structured3d')
    
    # List of expected files
    files = [f"structured3d_{str(i).zfill(2)}.tar.gz" for i in range(1, 16)]
    
    # Verify all files exist
    missing_files = [f for f in files if not (input_dir / f).exists()]
    if missing_files:
        print("Missing archive files:", missing_files)
        print("Please run download_structured3d.py first")
        return

    print("\nExtracting files...")
    
    # Extract all files in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for file in files:
            filepath = input_dir / file
            futures.append(executor.submit(extract_tar, filepath))
        
        # Wait for all extractions to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error during extraction: {e}")

    if not args.keep_archives:
        print("\nCleaning up downloaded archives...")
        # Remove the downloaded tar.gz files
        for file in files:
            filepath = input_dir / file
            filepath.unlink()

    print("\nExtraction complete!")
    print(f"Dataset is available in: {input_dir}")

if __name__ == '__main__':
    main() 