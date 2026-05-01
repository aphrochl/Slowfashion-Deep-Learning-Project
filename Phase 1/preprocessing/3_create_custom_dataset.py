
import os
import json
import shutil
import pathlib
from concurrent.futures import ProcessPoolExecutor
import time

# Configuration
SOURCE_IMAGES_DIR = r"C:\Users\achlapani\Downloads\DeepFashion2\data\ALL_Together"
SOURCE_ANNOS_DIR = r"C:\Users\achlapani\Downloads\DeepFashion2\data\raw\train\annos"
OUTPUT_DIR = r"C:\Users\achlapani\Downloads\DeepFashion2\data\DeepFashion2_Custom_Dataset"

# Mapping Logic
TARGET_MAPPING = {
    "Dresses": ["long sleeve dress", "short sleeve dress", "vest dress"],
    "Skirts": ["skirt"],
    "Outerwear": ["long sleeve outwear", "short sleeve outwear"]
}

# Invert mapping for easier lookup
FOLDER_TO_CATEGORY = {}
for category, folders in TARGET_MAPPING.items():
    for folder in folders:
        FOLDER_TO_CATEGORY[folder] = category

def transform_coordinates(points, x_offset, y_offset, x_scale, y_scale):
    """Transforms a list of [x, y, x, y...] coordinates."""
    transformed = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i+1]
        
        # Transform
        new_x = (x - x_offset) * x_scale
        new_y = (y - y_offset) * y_scale
        
        transformed.extend([new_x, new_y])
    return transformed

def transform_landmarks(landmarks, x_offset, y_offset, x_scale, y_scale):
    """Transforms landmarks [x, y, v, x, y, v...] preserving visibility 'v'."""
    transformed = []
    for i in range(0, len(landmarks), 3):
        x = landmarks[i]
        y = landmarks[i+1]
        v = landmarks[i+2]
        
        if x == 0 and y == 0:
             # Keep zero if it was zero (not visible/present)
             new_x, new_y = 0, 0
        else:
            new_x = (x - x_offset) * x_scale
            new_y = (y - y_offset) * y_scale
        
        transformed.extend([new_x, new_y, v])
    return transformed

def process_file(file_path):
    try:
        path = pathlib.Path(file_path)
        filename = path.name # e.g. 000332_item1.jpg
        parent_folder = path.parent.name # e.g. vest
        
        # Check if this folder is in our target mapping
        target_category = FOLDER_TO_CATEGORY.get(parent_folder)
        if not target_category:
            return None # Skip this file
            
        # Parse filename
        # Expecting format: {image_id}_{item_id}.jpg
        # But image_id can contain underscores? DeepFashion2 IDs are usually digits.
        # Let's assume the last part is item_id and the rest is image_id
        name_stem = path.stem # 000332_item1
        parts = name_stem.split('_')
        item_id = parts[-1] # item1
        image_id = "_".join(parts[:-1]) # 000332
        
        # Load Original Annotation
        anno_path = pathlib.Path(SOURCE_ANNOS_DIR) / f"{image_id}.json"
        
        if not anno_path.exists():
            return f"Error: Annotation not found for {filename}"
            
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)
            
        item_data = anno_data.get(item_id)
        if not item_data:
            return f"Error: Item {item_id} not found in {image_id}.json"
            
        # Get Original Bounding Box to calculate scale
        bbox = item_data.get('bounding_box') # [x1, y1, x2, y2]
        if not bbox:
            return f"Error: No bbox for {filename}"
            
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            return f"Error: Invalid bbox for {filename}"
            
        # Calculate Scale Factors (224x224 target)
        scale_x = 224.0 / w
        scale_y = 224.0 / h
        
        # Create New Item Data
        new_item = item_data.copy()
        
        # Transform Segmentation
        if 'segmentation' in new_item:
            new_segs = []
            for poly in new_item['segmentation']:
                new_segs.append(transform_coordinates(poly, x1, y1, scale_x, scale_y))
            new_item['segmentation'] = new_segs
            
        # Transform Landmarks
        if 'landmarks' in new_item:
            new_item['landmarks'] = transform_landmarks(new_item['landmarks'], x1, y1, scale_x, scale_y)
            
        # Transform BBox
        # Since the image IS the crop, the bbox is the full image
        new_item['bounding_box'] = [0, 0, 224, 224]
        
        # Update Category Name
        new_item['category_name'] = target_category
        # Note: We are NOT changing category_id automatically as it needs a consistent map.
        # User didn't strictly ask for it, but 'category_name' is updated.
        
        # Construct Output JSON Structure based on user request example
        # User example: {"item2": {...}, "source": "...", "pair_id": ...}
        # We will wrap our single item in a similar structure
        output_json = {
            "source": anno_data.get("source", "unknown"),
            "pair_id": anno_data.get("pair_id", 0),
            item_id: new_item  # e.g. "item1": { ... }
        }
        
        # Define Output Paths
        dest_dir = pathlib.Path(OUTPUT_DIR) / target_category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_img_path = dest_dir / filename
        dest_json_path = dest_dir / f"{name_stem}.json"
        
        # Copy Image
        shutil.copy2(path, dest_img_path)
        
        # Save JSON
        with open(dest_json_path, 'w') as f:
            json.dump(output_json, f)
            
        return 1 # Success
        
    except Exception as e:
        return f"Exception file {file_path}: {e}"

def main():
    print("Starting Custom Dataset Generation...")
    print(f"Source: {SOURCE_IMAGES_DIR}")
    print(f"Annos: {SOURCE_ANNOS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print("Source images directory not found!")
        return

    # Gather all candidate files
    all_files = []
    source_path = pathlib.Path(SOURCE_IMAGES_DIR)
    
    # Iterate known subfolders to save time
    for subfolder in FOLDER_TO_CATEGORY.keys():
        folder_path = source_path / subfolder
        if folder_path.exists():
            print(f"Scanning {subfolder}...")
            files = list(folder_path.glob("*.jpg"))
            all_files.extend(files)
        else:
            print(f"Warning: Subfolder {subfolder} not found in source.")
            
    total_files = len(all_files)
    print(f"Found {total_files} images to process.")
    
    max_workers = max(1, os.cpu_count() - 1)
    
    count_success = 0
    count_errors = 0
    errors = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        
        for i, future in enumerate(futures):
            result = future.result()
            if result == 1:
                count_success += 1
            else:
                count_errors += 1
                if result: errors.append(result)
                
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{total_files}...")
                
    end_time = time.time()
    print(f"Done in {end_time - start_time:.2f} seconds.")
    print(f"Success: {count_success}")
    print(f"Errors: {count_errors}")
    
    if errors:
        print("Sample errors:")
        for e in errors[:10]:
            print(e)

if __name__ == "__main__":
    main()
