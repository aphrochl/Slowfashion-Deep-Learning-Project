import zipfile
import os
from pathlib import Path
import time

# Configuration
ZIP_PASSWORD = b'2019Deepfashion2**'
REPORT_INTERVAL = 100 # Print every N files

def unzip_reliable():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    zip_path = base_dir / 'train.zip'
    extract_root = base_dir / 'data' / 'raw'

    print(f"Zip: {zip_path}")
    print(f"Target: {extract_root}")

    if not zip_path.exists():
        print(f"Error: {zip_path} not found.")
        return

    extract_root.mkdir(parents=True, exist_ok=True)

    print("Reading zip content structure... (this may take a moment)")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.setpassword(ZIP_PASSWORD)
            all_members = z.infolist()
            
            # Filter for files only
            files_to_extract = [f for f in all_members if not f.is_dir()]
            total_files = len(files_to_extract)
            print(f"Total files in archive: {total_files}")

            # Smart Resume: Check what's already there
            print("Checking existing files to resume...")
            missing_files = []
            
            # Optimization: Pre-scan directory if possible, but strict check is safer
            # For 191k files, os.path.exists is okay, or we can list dir once.
            # Let's trust pathlib.exists for simplicity and reliability.
            
            count_exists = 0
            for f in files_to_extract:
                target = extract_root / f.filename
                if target.exists() and target.stat().st_size == f.file_size:
                    count_exists += 1
                else:
                    missing_files.append(f)
            
            print(f"Found {count_exists} existing valid files.")
            print(f"Remaining to extract: {len(missing_files)}")
            
            if not missing_files:
                print("All files extracted.")
                return

            print("Starting extraction...")
            start_time = time.time()
            extracted_count = 0
            
            for i, file_info in enumerate(missing_files):
                z.extract(file_info, extract_root)
                extracted_count += 1
                
                if extracted_count % REPORT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = extracted_count / elapsed
                    remaining = len(missing_files) - extracted_count
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {extracted_count}/{len(missing_files)} ({extracted_count/len(missing_files)*100:.1f}%) - Rate: {rate:.1f} files/s - ETA: {eta/60:.1f} min")

    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    unzip_reliable()
