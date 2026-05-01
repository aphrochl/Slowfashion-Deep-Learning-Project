import torch
from pathlib import Path
import numpy as np

def verify():
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / 'data' / 'train' / 'data.pt'
    
    if not train_path.exists():
        print("Train file not found.")
        return

    print("Loading sorted train data...")
    data = torch.load(train_path)
    y_sub = data['targets_sub']
    classes = data['classes_sub']
    
    unique, counts = np.unique(y_sub, return_counts=True)
    
    print(f"Total Train Samples: {len(y_sub)}")
    print("Class Distribution:")
    for label, count in zip(unique, counts):
        print(f"  {classes[label]}: {count}")

if __name__ == "__main__":
    verify()
