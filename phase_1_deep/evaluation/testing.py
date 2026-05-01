
import os
import glob
import json
import random
import pathlib
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ----------------- CONFIG -----------------
# Dynamically find the data folder relative to this script
# Layout: slowfashion/evaluation/testing.py -> data is at slowfashion/data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'slowfashion_custom_model_final.pth'

# MAPPING DEFINITIONS (Must match train.py)
SUB_MAPPING = {
    "Dresses": 0,
    "Skirts": 3,
    "Outerwear": 6
}
GROUP_MAPPING = {
    "Dresses": 2,
    "Skirts": 2,
    "Outerwear": 2
}

# ----------------- DATASET -----------------
class DeepFashionLazyDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.transform = transform
        
        print(f"Initializing Test Dataset from: {root_dirs}")
        
        for d in root_dirs:
            path_obj = pathlib.Path(d)
            if not path_obj.exists():
                print(f"WARNING: Directory {d} does not exist!")
                continue

            jsons = list(path_obj.rglob("*.json"))
            print(f"  Found {len(jsons)} items in {d}")
            
            for j in jsons:
                try:
                    category = self._read_label(j)
                    if category in SUB_MAPPING:
                        img_path = j.with_suffix('.jpg')
                        if img_path.exists():
                            sub_label = SUB_MAPPING[category]
                            group_label = GROUP_MAPPING[category]
                            self.samples.append((str(img_path), sub_label, group_label))
                except Exception as e:
                    pass
        
        print(f"Total Test samples loaded: {len(self.samples)}")

    def _read_label(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if isinstance(value, dict) and "category_name" in value:
                    return value["category_name"]
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sub_label, group_label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, sub_label, group_label

# ----------------- MODEL -----------------
class DualOutputResNet(nn.Module):
    def __init__(self, num_groups=3, num_sub=9):
        super(DualOutputResNet, self).__init__()
        
        # We only need architecture definition to load weights
        # We can set pretrained=False since we are loading our own weights
        self.backbone = models.resnet18(pretrained=False)
            
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 
        
        self.group_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_groups)
        )
        
        self.sub_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_sub)
        )

    def forward(self, x):
        features = self.backbone(x)
        group_out = self.group_head(features)
        sub_out = self.sub_head(features)
        return group_out, sub_out

# ----------------- TESTING -----------------
def test_model():
    print(f"Starting Testing on {DEVICE}")
    
    # 1. Load Data
    test_dir = DATA_DIR / 'test'
    if not test_dir.exists():
        print(f"Critical Error: 'test' directory not found at {test_dir}")
        print("Please ensure your dataset is unzipped cleanly into data/train and data/test")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DeepFashionLazyDataset([test_dir], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(test_dataset) == 0:
        print("No test data found. Exiting.")
        return

    # 2. Load Model
    model = DualOutputResNet(num_groups=3, num_sub=9).to(DEVICE)
    
    # Locate model file (check current, training/, or root)
    candidates = [
        BASE_DIR / MODEL_PATH,
        BASE_DIR / 'training' / MODEL_PATH,
        Path(MODEL_PATH)
    ]
    
    model_file = None
    for p in candidates:
        if p.exists():
            model_file = p
            break
            
    if model_file is None:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Expected locations:")
        for p in candidates: print(f" - {p}")
        return

    print(f"Loading weights from {model_file}...")
    try:
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model.eval()
    
    # 3. Evaluate
    all_preds_sub = []
    all_targets_sub = []
    all_preds_group = []
    all_targets_group = []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, sub_labels, group_labels in test_loader:
            inputs = inputs.to(DEVICE)
            
            out_group, out_sub = model(inputs)
            
            _, pred_sub = out_sub.max(1)
            _, pred_group = out_group.max(1)
            
            all_preds_sub.extend(pred_sub.cpu().numpy())
            all_targets_sub.extend(sub_labels.numpy())
            
            all_preds_group.extend(pred_group.cpu().numpy())
            all_targets_group.extend(group_labels.numpy())
            
    # 4. Metrics
    print("\n" + "="*40)
    print("       FINAL TEST RESULTS       ")
    print("="*40)
    
    # Subcategory Metrics
    acc_sub = accuracy_score(all_targets_sub, all_preds_sub)
    f1_sub = f1_score(all_targets_sub, all_preds_sub, average='macro', zero_division=0)
    
    print(f"Subcategory Accuracy:   {acc_sub:.4f}")
    print(f"Subcategory F1 (Macro): {f1_sub:.4f}")
    print("-" * 20)
    
    # Group Metrics
    acc_group = accuracy_score(all_targets_group, all_preds_group)
    print(f"Group Accuracy:         {acc_group:.4f}")
    
    print("\n[Detailed Subcategory Report]")
    # Map back indices to names for report
    # 0->Dresses, 3->Skirts, 6->Outerwear
    # BUT classification_report expects contiguous indices if we provide target_names?
    # No, we can provide labels=[0, 3, 6]
    
    target_names = ["Dresses", "Skirts", "Outerwear"]
    labels = [0, 3, 6]
    
    try:
        print(classification_report(all_targets_sub, all_preds_sub, labels=labels, target_names=target_names))
    except:
        print("Could not generate classification report (possibly missing classes in test set).")
        print("Raw unique targets:", np.unique(all_targets_sub))

    print("="*40)

if __name__ == "__main__":
    test_model()
