
import os
import shutil
import pathlib
import time

SOURCE_DIR = r"C:\Users\achlapani\Downloads\DeepFashion2\data\DeepFashion2_Custom_Dataset"
OUTPUT_DIR = r"C:\Users\achlapani\Downloads\DeepFashion2\data\custom_data2"

def main():
    print(f"Flattening dataset...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    
    source_path = pathlib.Path(SOURCE_DIR)
    dest_path = pathlib.Path(OUTPUT_DIR)
    
    if not source_path.exists():
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        return

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Gather all files recursively
    # We want .jpg and .json files
    print("Scanning source files...")
    all_files = list(source_path.rglob("*"))
    files_to_copy = [f for f in all_files if f.is_file()]
    
    total_files = len(files_to_copy)
    print(f"Found {total_files} files to copy.")
    
    start_time = time.time()
    count = 0
    errors = 0
    
    for i, file_path in enumerate(files_to_copy):
        try:
            # Flatten: Copy directly to OUTPUT_DIR, preserving filename
            shutil.copy2(file_path, dest_path / file_path.name)
            count += 1
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")
            errors += 1
            
        if i % 1000 == 0 and i > 0:
            print(f"Copied {i}/{total_files}...")

    end_time = time.time()
    print(f"Done in {end_time - start_time:.2f} seconds.")
    print(f"Successfully copied: {count}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
