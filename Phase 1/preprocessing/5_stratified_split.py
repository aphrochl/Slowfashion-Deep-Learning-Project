
import os
import shutil
import json
import random
import pathlib
import collections
import concurrent.futures

# Configuration
DATA_DIR = r"C:\Users\achlapani\Downloads\DeepFashion2\data\custom_data2"
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
SEED = 42

def load_category(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Annotations are wrapped in "item1", "item2" etc inside the json, 
            # or sometimes key is "category_name" directly if previous script flattened it deeply?
            # Let's check the structure based on previous view_file:
            # {"source": "user", "pair_id": 2, "item1": { ... "category_name": "Dresses" }}
            
            # We need to find the item dictionary. It's usually the third key or start with "item"
            for key, value in data.items():
                if key.startswith("item") and isinstance(value, dict) and "category_name" in value:
                    return value["category_name"]
            
            # Fallback or error
            return "Unknown"
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def main():
    random.seed(SEED)
    data_path = pathlib.Path(DATA_DIR)
    
    print(f"Scanning {DATA_DIR}...")
    
    # Get all JSON files first (as they contain the labels)
    all_jsons = list(data_path.glob("*.json"))
    total_files = len(all_jsons)
    
    print(f"Found {total_files} items to split.")
    
    # Group by category
    category_map = collections.defaultdict(list)
    
    # Use parallel processing to read categories faster if many files
    print("Reading categories...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Map json path to category
        future_to_path = {executor.submit(load_category, p): p for p in all_jsons}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
            if i % 5000 == 0:
                print(f"Processed {i}/{total_files} annotations...")
                
            cat = future.result()
            p = future_to_path[future]
            if cat and cat != "Unknown":
                # Store the base name (without extension) or the full path stem
                # We need to move both .jpg and .json
                category_map[cat].append(p)
            else:
                print(f"Warning: Could not extract category from {p.name}")

    print("\nCategory Distribution:")
    for cat, items in category_map.items():
        print(f"  {cat}: {len(items)}")
        
    # Create split directories
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        split_dir.mkdir(exist_ok=True)
        print(f"Created {split_dir}")

    # Perform Split and Move
    print("\nPerforming stratified split and moving files...")
    moves_count = 0
    
    for cat, items in category_map.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * SPLIT_RATIOS['train'])
        n_val = int(n * SPLIT_RATIOS['val'])
        # Remainder goes to test to ensure sum is n
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train+n_val]
        test_items = items[n_train+n_val:]
        
        splits = {
            'train': train_items,
            'val': val_items,
            'test': test_items
        }
        
        for split_name, file_list in splits.items():
            target_dir = data_path / split_name
            for json_path in file_list:
                # Move JSON
                shutil.move(str(json_path), str(target_dir / json_path.name))
                
                # Move corresponding JPG
                jpg_path = json_path.with_suffix('.jpg')
                if jpg_path.exists():
                    shutil.move(str(jpg_path), str(target_dir / jpg_path.name))
                    moves_count += 1
                else:
                    print(f"Warning: JPG not found for {json_path.name}")

    print(f"\nDone! Moved {moves_count} pairs.")

if __name__ == "__main__":
    main()
