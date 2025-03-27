import subprocess
import os

def download_waymo_data(bucket_path, local_path):
    """
    Download data from Google Cloud Storage bucket to local path
    
    Args:
        bucket_path (str): GCS bucket path
        local_path (str): Local directory path where files will be downloaded
    """
    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Construct gsutil command
    cmd = f"gsutil -m cp -r {bucket_path} {local_path}"
    
    try:
        print(f"Downloading from {bucket_path} to {local_path}")
        subprocess.run(cmd, shell=True, check=True)
        print("Download completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")

def main():
    # Base bucket path for Waymo Open Dataset v1.4.3
    base_bucket = "gs://waymo_open_dataset_v_1_4_3/individual_files"
    
    # Define paths for training and validation sets
    train_path = f"{base_bucket}/training"
    val_path = f"{base_bucket}/validation"
    
    # Local directories where data will be saved
    local_base_dir = "./waymo_data"
    local_train_dir = os.path.join(local_base_dir, "training")
    local_val_dir = os.path.join(local_base_dir, "validation")
    
    # Download training data
    print("Downloading training data...")
    download_waymo_data(train_path, local_train_dir)
    
    # Download validation data
    print("Downloading validation data...")
    download_waymo_data(val_path, local_val_dir)

if __name__ == "__main__":
    main()