import torch
import numpy as np
from pathlib import Path

def check_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'data' / 'processed' / 'all_data.pt'
    
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return
        
    print(f"Loading {data_path}...")
    data = torch.load(data_path)
    targets = data['targets'].numpy()
    
    unique, counts = np.unique(targets, return_counts=True)
    mapping = {
        0: "Dresses",
        1: "High Heels",
        2: "Shoulder Bags",
        3: "Skirts",
        4: "Tote Bags",
        5: "Clutches",
        6: "Outerwear",
        7: "Boots",
        8: "Flats"
    }
    
    print("\nClass Distribution:")
    total = 0
    for label, count in zip(unique, counts):
        name = mapping.get(label, "Unknown")
        print(f"Class {label} ({name}): {count}")
        total += count
    print(f"\nTotal samples: {total}")

if __name__ == "__main__":
    check_data()
