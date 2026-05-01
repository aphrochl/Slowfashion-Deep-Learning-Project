import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data(val_size=0.1, test_size=0.1, seed=42):
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / 'data' / 'processed'
    data_path = processed_dir / 'all_data.pt'
    
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run 3_mapping.py first.")
        return

    print(f"Loading data from {data_path}...")
    data_dict = torch.load(data_path)
    X = data_dict['data']
    y_sub = data_dict['targets_sub']
    y_group = data_dict['targets_group']
    
    # We stratify by subcategory to ensure representation
    stratify_labels = y_sub
    
    # Stratified split: Train (90%) vs Test (10%)
    indices = np.arange(len(X))
    
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify_labels
    )
    
    splits = {
        'train': train_idx,
        'test': test_idx
    }
    
    dirs = {
        'train': base_dir / 'data' / 'train',
        'test': base_dir / 'data' / 'test'
    }
    
    # Clean up old val dir
    val_dir = base_dir / 'data' / 'val'
    if val_dir.exists():
        import shutil
        shutil.rmtree(val_dir)
        print(f"Removed old validation directory: {val_dir}")
    
    for split_name, idx in splits.items():
        X_split = X[idx]
        y_sub_split = y_sub[idx]
        y_group_split = y_group[idx]
        
        # Ensure directory exists
        dirs[split_name].mkdir(parents=True, exist_ok=True)
        
        output_path = dirs[split_name] / 'data.pt'
        torch.save({
            'data': X_split,
            'targets_sub': y_sub_split,
            'targets_group': y_group_split,
            'classes_sub': data_dict['classes_sub'],
            'classes_group': data_dict['classes_group']
        }, output_path)
        
        print(f"{split_name.capitalize()} set: {len(idx)} samples saved to {output_path}")

    # Verify Mappings in Train directly here
    print("\nVerifying Mappings in Train Set:")
    unique_sub, counts_sub = np.unique(y_sub[train_idx], return_counts=True)
    for label, count in zip(unique_sub, counts_sub):
        name = data_dict['classes_sub'][label]
        print(f"  {name}: {count}")

if __name__ == "__main__":
    split_data()
