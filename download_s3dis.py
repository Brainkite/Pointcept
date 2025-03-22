import os
import tarfile
from datasets import DownloadManager
from tqdm import tqdm

def download_and_unpack_tar(url, output_folder):
    """Downloads a tar.gz file and unpacks it into a specified folder with progress."""

    download_manager = DownloadManager()
    local_path = download_manager.download(url)

    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    try:
        with tarfile.open(local_path, "r:gz") as tar:
            members = tar.getmembers()
            total_size = sum(member.size for member in members)
            extracted_size = 0

            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, path=output_folder)
                    extracted_size += member.size
                    pbar.update(member.size)

        print(f"Successfully unpacked {url} into {output_folder}")
    except tarfile.ReadError:
        print(f"Error: Could not unpack {url}. Invalid tar.gz file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
url = "https://huggingface.co/datasets/Pointcept/s3dis-compressed/resolve/main/s3dis.tar.gz"
output_folder = "./data/s3dis"  # Choose your desired output folder

download_and_unpack_tar(url, output_folder)