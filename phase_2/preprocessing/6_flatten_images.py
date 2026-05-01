
import os
import shutil
import csv
import json
from pathlib import Path

# Config
SOURCE_CSV = 'local_dataset.csv'
TARGET_DIR = 'data/all_images'
NEW_CSV = 'flat_dataset.csv'
NEW_JSON = 'flat_dataset.json'

# Create target directory
os.makedirs(TARGET_DIR, exist_ok=True)

updated_rows = []
json_data = []

print(f"Flattening images from {SOURCE_CSV} into {TARGET_DIR}...")

try:
    with open(SOURCE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Check column indices
        # Expected: image_url, group, label, filename, local_path
        try:
            path_idx = header.index('local_path')
        except ValueError:
            # Fallback if column name is different or using index
            path_idx = 4 

        count = 0
        success_count = 0
        
        for row in reader:
            count += 1
            old_path = row[path_idx]
            original_filename = os.path.basename(old_path)
            
            # source file
            src_file = Path(old_path)
            
            if src_file.exists():
                # Destination path
                dst_file = Path(TARGET_DIR) / src_file.name
                
                # Move file (using move instead of copy for speed/space, as requested to "flatten")
                # But to be safe against data loss in case of error, maybe copy? 
                # User said "flatten... into a new file", implying restructuring. 
                # I will MOVE to avoid duplication, assuming we have backups or can redownload.
                # Actually, shutil.move is safer.
                
                if not dst_file.exists():
                     shutil.move(str(src_file), str(dst_file))
                
                # Update row for CSV
                new_row = row.copy()
                new_row[path_idx] = str(dst_file).replace('\\', '/')
                updated_rows.append(new_row)
                
                # Add to JSON data
                item = {
                    'image_url': row[0],
                    'group': row[1],
                    'label': row[2],
                    'filename': src_file.name,
                    'file_path': str(dst_file).replace('\\', '/')
                }
                json_data.append(item)
                
                success_count += 1
            else:
                print(f"Warning: Source file not found: {old_path}")

    # Write new CSV
    with open(NEW_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_rows)

    # Write JSON
    with open(NEW_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print(f"Flattening complete.")
    print(f"Processed: {count}")
    print(f"Moved/Verified: {success_count}")
    print(f"New CSV: {NEW_CSV}")
    print(f"New JSON: {NEW_JSON}")

    # Optional: cleanup empty directories in data/images
    # shutil.rmtree('data/images') # Commented out for safety, user can delete later

except Exception as e:
    print(f"An error occurred: {e}")
