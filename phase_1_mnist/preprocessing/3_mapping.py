import gzip
import os
import numpy as np
import torch
from pathlib import Path

# Constants for file names
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

# --- LABELS SCHEMA ---
# Groups
GROUPS = {0: 'Bags', 1: 'Shoes', 2: 'Clothing'}

# Subcategories (Target Labels)
SUBCATEGORIES = {
    0: 'Dresses',
    1: 'High Heels',
    2: 'Shoulder Bags',
    3: 'Skirts',
    4: 'Tote Bags',
    5: 'Clutches',
    6: 'Outerwear',
    7: 'Boots',
    8: 'Flats'
}

# Subcategory -> Group Mapping
SUB_TO_GROUP = {
    0: 2, # Dresses -> Clothing
    1: 1, # High Heels -> Shoes
    2: 0, # Shoulder Bags -> Bags
    3: 2, # Skirts -> Clothing
    4: 0, # Tote Bags -> Bags
    5: 0, # Clutches -> Bags
    6: 2, # Outerwear -> Clothing
    7: 1, # Boots -> Shoes
    8: 1  # Flats -> Shoes
}

# --- RAW MAPPING ---
# FashionMNIST -> Subcategory ID
# 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 
# 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
RAW_TO_SUB = {
    0: 6,  # T-shirt/top -> Outerwear
    1: -1, # Trouser -> Exclude (No matching subcategory)
    2: 6,  # Pullover -> Outerwear
    3: 0,  # Dress -> Dresses
    4: 6,  # Coat -> Outerwear
    5: 8,  # Sandal -> Flats
    6: 6,  # Shirt -> Outerwear
    7: 8,  # Sneaker -> Flats
    8: 2,  # Bag -> Shoulder Bags (Map all bags here for now as generic)
    9: 7   # Ankle boot -> Boots
}

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def process_data():
    base_dir = Path(__file__).resolve().parent.parent # Slowfashion/
    
    # Define potential raw data paths
    # 1. Local relative path
    raw_dir = base_dir / 'data' / 'raw'

    if not raw_dir.exists():
        # Fallback check or Just error
        print(f"Error: Raw data not found at {raw_dir}")
        return

    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load separate files
    print("Loading raw data...")
    try:
        train_X = load_mnist_images(raw_dir / TRAIN_IMAGES)
        train_y = load_mnist_labels(raw_dir / TRAIN_LABELS)
        test_X = load_mnist_images(raw_dir / TEST_IMAGES)
        test_y = load_mnist_labels(raw_dir / TEST_LABELS)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    # Concatenate
    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)
    
    # Apply Mapping
    print("Applying tiered mapping...")
    sub_labels = np.array([RAW_TO_SUB.get(label, -1) for label in y])
    
    # Filter valid
    mask = sub_labels != -1
    X_filtered = X[mask]
    sub_filtered = sub_labels[mask]
    
    # Generate Group Labels
    group_filtered = np.array([SUB_TO_GROUP[sub] for sub in sub_filtered])
    
    print(f"Total valid samples: {len(sub_filtered)}")
    
    # Check counts
    unique_sub, counts_sub = np.unique(sub_filtered, return_counts=True)
    print("\nSubcategory Distribution:")
    for label, count in zip(unique_sub, counts_sub):
        print(f"  {SUBCATEGORIES[label]}: {count}")
        
    unique_group, counts_group = np.unique(group_filtered, return_counts=True)
    print("\nGroup Distribution:")
    for label, count in zip(unique_group, counts_group):
        print(f"  {GROUPS[label]}: {count}")

    # Normalize (0-1)
    print("\nNormalizing data...")
    X_filtered = X_filtered.astype(np.float32) / 255.0
    
    # Convert to Tensor
    X_tensor = torch.from_numpy(X_filtered).unsqueeze(1) # [N, 1, 28, 28]
    y_sub_tensor = torch.from_numpy(sub_filtered).long()
    y_group_tensor = torch.from_numpy(group_filtered).long()
    
    # Save dictionary with both targets
    output_path = processed_dir / 'all_data.pt'
    torch.save({
        'data': X_tensor,
        'targets_sub': y_sub_tensor,
        'targets_group': y_group_tensor,
        'classes_sub': SUBCATEGORIES,
        'classes_group': GROUPS
    }, output_path)
    print(f"Saved processed tiered data to {output_path}")

if __name__ == "__main__":
    process_data()
