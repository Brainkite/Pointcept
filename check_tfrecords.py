""":
Script to check Waymo Open Dataset TFRecord files for parsing errors.
"""

import argparse
import os
import glob
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from google.protobuf.message import DecodeError

# Suppress TensorFlow logging (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR') # Suppress INFO and WARNING messages


def check_tfrecord_file(file_path):
    """
    Checks a single TFRecord file for frame parsing errors.

    Args:
        file_path (str): Path to the TFRecord file.

    Returns:
        bool: True if the file is processed without parsing errors, False otherwise.
    """
    print(f"Checking: {file_path}...")
    try:
        dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        for i, data in enumerate(dataset):
            try:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
            except (DecodeError, Exception) as e: # Catch DecodeError and other potential issues
                print(f"  \x1b[91mERROR\x1b[0m parsing record {i} in file: {file_path}")
                print(f"    Error details: {e}")
                return False
        print(f"  \x1b[92mOK\x1b[0m - Parsed all records successfully.")
        return True
    except tf.errors.DataLossError as e:
        print(f"  \x1b[91mERROR\x1b[0m reading file (DataLossError): {file_path}")
        print(f"    Error details: {e}")
        return False
    except Exception as e:
        print(f"  \x1b[91mERROR\x1b[0m opening or reading file: {file_path}")
        print(f"    Error details: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check Waymo TFRecord files for parsing errors."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Directory containing TFRecord files (searches recursively).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Directory not found: {args.dataset_dir}")
        exit(1)

    print(f"Searching for .tfrecord files in: {args.dataset_dir}")
    # Use recursive glob to find all tfrecord files
    tfrecord_files = glob.glob(os.path.join(args.dataset_dir, "**", "*.tfrecord"), recursive=True)

    if not tfrecord_files:
        print("No .tfrecord files found.")
        exit(0)

    print(f"Found {len(tfrecord_files)} TFRecord files. Starting check...")
    failed_files = []

    for file_path in tfrecord_files:
        if not check_tfrecord_file(file_path):
            failed_files.append(file_path)

    print("\n--- Check Complete ---")
    if failed_files:
        print(f"\x1b[91m{len(failed_files)} file(s) failed parsing:\x1b[0m")
        for f in failed_files:
            print(f"  - {f}")
    else:
        print("\x1b[92mAll files parsed successfully!\x1b[0m") 