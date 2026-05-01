import zipfile
import os
from pathlib import Path
import sys

def unzip_train_data():
    # Base paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent # DeepFashion2 root
    zip_path = base_dir / 'train.zip'
    extract_to = base_dir / 'data' / 'raw'

    print(f"Base Directory: {base_dir}")
    print(f"Zip File: {zip_path}")
    print(f"Extract Target: {extract_to}")

    if not zip_path.exists():
        print(f"Error: {zip_path} not found.")
        return

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check first few files to see structure
            top_level = {x.split('/')[0] for x in zip_ref.namelist()[:20]}
            print(f"Archive content structure: {top_level}")
            
            # Extract
            print("Extracting with password...")
            zip_ref.extractall(extract_to, pwd=b'2019Deepfashion2**')
            
        print(f"Successfully extracted to {extract_to}")
    except Exception as e:
        print(f"Error unzipping: {e}")
        sys.exit(1)

if __name__ == "__main__":
    unzip_train_data()
