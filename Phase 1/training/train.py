
import os
import glob
import json
import random
import pathlib
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models

# ----------------- CONFIG -----------------
# Dynamically find the data folder relative to this script
# Layout: server/train.py -> data is at server/data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

BATCH_SIZE = 32
EPOCHS = 3
LR = 0.001
N_FOLDS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# MAPPING DEFINITIONS
# Global Subcategory IDs (9 Total)
SUB_MAPPING = {
    "Dresses": 0,
    "Skirts": 3,
    "Outerwear": 6
}

# Global Group IDs (3 Total)
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
        
        print(f"Initializing Dataset from: {root_dirs}")
        
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
        
        print(f"Total samples loaded: {len(self.samples)}")

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
        try:
            image = Image.open(img_path).convert('RGB')
        except:
             # Fallback for corrupt images? return a black image or error out. 
             # For now let it error to be visible
             raise 
        
        if self.transform:
            image = self.transform(image)
            
        return image, sub_label, group_label

# ----------------- MODEL -----------------
class DualOutputResNet(nn.Module):
    def __init__(self, num_groups=3, num_sub=9):
        super(DualOutputResNet, self).__init__()
        
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
            print("SUCCESS: Loaded pretrained ResNet18 weights (ImageNet) from torchvision.")
        except Exception as e:
            print(f"NOTE: New weights API failed ({e}), trying legacy 'pretrained=True'...")
            try:
                self.backbone = models.resnet18(pretrained=True)
                print("SUCCESS: Loaded pretrained ResNet18 weights (ImageNet) using legacy API.")
            except Exception as e2:
                print(f"WARNING: Could not load pretrained weights! Internet might be down. Error: {e2}")
                print("FALLBACK: Using Random Initialization (Not Transfer Learning).")
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

# ----------------- TRAIN UTILS -----------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_sub = 0
    correct_group = 0
    total = 0
    
    for inputs, sub_labels, group_labels in loader:
        inputs = inputs.to(DEVICE)
        sub_labels = sub_labels.to(DEVICE)
        group_labels = group_labels.to(DEVICE)
        
        optimizer.zero_grad()
        out_group, out_sub = model(inputs)
        
        loss_group = criterion(out_group, group_labels)
        loss_sub = criterion(out_sub, sub_labels)
        loss = loss_group + loss_sub
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _,  pred_sub = out_sub.max(1)
        _, pred_group = out_group.max(1)
        
        total += inputs.size(0)
        correct_sub += pred_sub.eq(sub_labels).sum().item()
        correct_group += pred_group.eq(group_labels).sum().item()
        
    return running_loss / total, correct_sub / total, correct_group / total

def evaluate_metrics(model, loader):
    model.eval()
    all_preds_sub = []
    all_targets_sub = []
    all_preds_group = []
    all_targets_group = []
    
    with torch.no_grad():
        for inputs, sub_labels, group_labels in loader:
            inputs = inputs.to(DEVICE)
            sub_labels = sub_labels.to(DEVICE)
            group_labels = group_labels.to(DEVICE)
            
            out_group, out_sub = model(inputs)
            
            _, pred_sub = out_sub.max(1)
            _, pred_group = out_group.max(1)
            
            all_preds_sub.extend(pred_sub.cpu().numpy())
            all_targets_sub.extend(sub_labels.cpu().numpy())
            
            all_preds_group.extend(pred_group.cpu().numpy())
            all_targets_group.extend(group_labels.cpu().numpy())
            
    # Metrics
    acc_sub = accuracy_score(all_targets_sub, all_preds_sub)
    f1_sub = f1_score(all_targets_sub, all_preds_sub, average='macro', zero_division=0)
    
    acc_group = accuracy_score(all_targets_group, all_preds_group)
    
    cm_sub = confusion_matrix(all_targets_sub, all_preds_sub, labels=list(range(9)))
    per_class_acc = np.divide(cm_sub.diagonal(), cm_sub.sum(axis=1), out=np.zeros_like(cm_sub.diagonal(), dtype=float), where=cm_sub.sum(axis=1)!=0)
    
    cm_group = confusion_matrix(all_targets_group, all_preds_group, labels=list(range(3)))
    per_group_acc = np.divide(cm_group.diagonal(), cm_group.sum(axis=1), out=np.zeros_like(cm_group.diagonal(), dtype=float), where=cm_group.sum(axis=1)!=0)

    return acc_sub, f1_sub, acc_group, per_class_acc, per_group_acc

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print(f"Using device: {DEVICE}")
    print(f"Data Directory being used: {DATA_DIR}")
    
    if not DATA_DIR.exists():
         print("CRITICAL ERROR: Data directory not found. Please verify paths.")
         return

    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = DATA_DIR / 'train'
    full_dataset = DeepFashionLazyDataset([train_dir], transform=transform)
    
    targets = [s[1] for s in full_dataset.samples]
    targets = np.array(targets)
    
    if len(targets) == 0:
        print("No data found! Exiting.")
        return

    # 2. Setup Resume Capability
    out_file = 'cv_results_custom.json'
    
    # Default Structure
    cv_results = {
        "fold_accuracies": [], # Track completed fold accuracies here
        "mean_accuracy": 0.0,
        "std_accuracy": 0.0,
        "per_class_accuracy": { str(i): [] for i in range(9) },
        "per_group_accuracy": { str(i): [] for i in range(3) }
    }
    
    folds_completed = 0
    if os.path.exists(out_file):
        try:
            with open(out_file, 'r') as f:
                loaded_results = json.load(f)
                # Check compatibility
                if "fold_accuracies" in loaded_results:
                    cv_results = loaded_results
                    folds_completed = len(cv_results["fold_accuracies"])
                    print(f"\n[RESUME] Found existing results. Resuming from Fold {folds_completed + 1}...")
        except Exception as e:
            print(f"[RESUME WARNING] Could not load {out_file}: {e}. Starting from scratch.")

    # 3. Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    print(f"\nStarting {N_FOLDS}-Fold Cross Validation...")
    
    # Use split on the full targets
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        
        # SKIP LOGIC
        if fold < folds_completed:
            print(f"Skipping Fold {fold+1} (Already Completed)")
            continue
            
        print(f"\nFold {fold+1}/{N_FOLDS}")
        
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=0)
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, num_workers=0)
        
        # Init Model
        model = DualOutputResNet(num_groups=3, num_sub=9).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        best_val_acc = 0.0
        best_metrics = None
        
        for epoch in range(EPOCHS):
            train_loss, train_acc_sub, train_acc_group = train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc_sub, val_f1_sub, val_acc_group, per_class_acc, per_group_acc = evaluate_metrics(model, val_loader)
            
            print(f"  Ep {epoch+1}/{EPOCHS}: Loss {train_loss:.4f} | Sub Acc {train_acc_sub:.4f}/{val_acc_sub:.4f}")
            
            if val_acc_sub > best_val_acc:
                best_val_acc = val_acc_sub
                best_metrics = {
                    "acc": val_acc_sub,
                    "per_class": per_class_acc,
                    "per_group": per_group_acc
                }

        # End of Fold - Save Results Immediately
        if best_metrics:
            # 1. Update In-Memory Results
            cv_results["fold_accuracies"].append(best_metrics['acc'])
            
            for i, acc_val in enumerate(best_metrics['per_class']):
                cv_results["per_class_accuracy"][str(i)].append(float(acc_val))
            
            for i, acc_val in enumerate(best_metrics['per_group']):
                cv_results["per_group_accuracy"][str(i)].append(float(acc_val))
                
            # Recalculate mean/std
            cv_results["mean_accuracy"] = float(np.mean(cv_results["fold_accuracies"]))
            cv_results["std_accuracy"] = float(np.std(cv_results["fold_accuracies"]))
            
            # 2. Write JSON to Disk
            with open(out_file, 'w') as f:
                json.dump(cv_results, f, indent=4)
            print(f"  -> Saved cumulative metrics to {out_file}")
            
            # 3. Save Model Checkpoint
            model_name = f'slowfashion_model_fold_{fold+1}.pth'
            torch.save(model.state_dict(), model_name)
            print(f"  -> Saved checkpoint: {model_name}")

            # 4. Also update 'final' reference?
            # User might want the 'last good model' easily accessible.
            torch.save(model.state_dict(), 'slowfashion_custom_model_final.pth')

    print("\n--- Cross Validation Results ---")
    print(f"Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    print(f"All Results saved to {out_file}")

if __name__ == "__main__":
    main()
