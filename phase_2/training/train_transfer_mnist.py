import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import json
import time
from pathlib import Path
from torchvision import transforms
from dataset import SlowFashionCsvDataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10
N_REPEATS = 1  # Updated to match DeepFashion (Standard 5-Fold)
K_FOLDS = 5    # Updated to match DeepFashion
# Model path logic: Models are in PROJECT_ROOT, script is in PROJECT_ROOT/training
ROOT_DIR = Path(__file__).resolve().parent.parent # Project Root
MODEL_PATH = ROOT_DIR / 'mnist_model.pth'
DATA_CSV = ROOT_DIR / 'data' / 'metadata' / 'train.csv'
RESULTS_DIR = ROOT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# --- Model Definition (Updated for Broader Groups) ---
class DualOutputCNN(nn.Module):
    def __init__(self, num_groups=6, num_sub=10):
        super(DualOutputCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.feature_size = 64 * 7 * 7
        self.group_head = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_groups)
        )
        self.sub_head = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_sub)
        )

    def forward(self, x):
        features = self.features(x)
        group_out = self.group_head(features)
        sub_out = self.sub_head(features)
        return group_out, sub_out

def train_cv():
    print(f"Starting MNIST Transfer Learning CV (Broader Groups) on {DEVICE}")
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    dataset = SlowFashionCsvDataset(
        csv_file=DATA_CSV,
        root_dir=ROOT_DIR,
        transform=transform,
        grayscale=True
    )
    
    from dataset import SUB_TO_IDX
    y_stratify = []
    for label in dataset.data['label']:
        if label in SUB_TO_IDX:
            y_stratify.append(SUB_TO_IDX[label])
        else:
            y_stratify.append(9) 
            
    print(f"Total Samples: {len(dataset)}")
    
    results = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': [],
        'per_class_accuracy': {i: [] for i in range(10)}, 
        'per_group_accuracy': {i: [] for i in range(6)}   
    }
    
    state_dict_to_load = None
    if MODEL_PATH.exists():
        print(f"Loading weights from {MODEL_PATH}...")
        raw_state = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict_to_load = {k: v for k, v in raw_state.items() if k.startswith('features')}
        print(f"Transferred {len(state_dict_to_load)}/{len(raw_state)} layers (Backbone only). Heads initialized from scratch.")
    else:
        print(f"Warning: {MODEL_PATH} not found. Training from scratch.")

    for repeat in range(N_REPEATS):
        print(f"\n--- Repeat {repeat+1}/{N_REPEATS} ---")
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42 + repeat)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_stratify)), y_stratify)):
            print(f"  Fold {fold+1}/{K_FOLDS}", end='... ')
            
            model = DualOutputCNN(num_groups=6, num_sub=10).to(DEVICE)
            if state_dict_to_load:
                model.load_state_dict(state_dict_to_load, strict=False)
            
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()
            
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)
            
            for epoch in range(EPOCHS):
                model.train()
                running_loss = 0.0
                correct_sub = 0
                correct_group = 0
                total = 0
                
                for X_b, y_g_b, y_s_b in train_loader:
                    X_b, y_g_b, y_s_b = X_b.to(DEVICE), y_g_b.to(DEVICE), y_s_b.to(DEVICE)
                    optimizer.zero_grad()
                    out_g, out_s = model(X_b)
                    loss = criterion(out_g, y_g_b) + criterion(out_s, y_s_b)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * X_b.size(0)
                    _, pred_s = torch.max(out_s, 1)
                    _, pred_g = torch.max(out_g, 1)
                    correct_sub += (pred_s == y_s_b).sum().item()
                    correct_group += (pred_g == y_g_b).sum().item()
                    total += y_s_b.size(0)
                
                epoch_loss = running_loss / total
                train_sub_acc = correct_sub / total
                train_group_acc = correct_group / total
                
                # Validation Per Epoch
                model.eval()
                val_preds_sub = []
                val_targets_sub = []
                val_preds_group = []
                val_targets_group = []
                
                with torch.no_grad():
                    for X_b, y_g_b, y_s_b in val_loader:
                        X_b = X_b.to(DEVICE)
                        out_g, out_s = model(X_b)
                        
                        _, ps = torch.max(out_s, 1)
                        _, pg = torch.max(out_g, 1)
                        
                        val_preds_sub.extend(ps.cpu().numpy())
                        val_targets_sub.extend(y_s_b.numpy())
                        val_preds_group.extend(pg.cpu().numpy())
                        val_targets_group.extend(y_g_b.numpy())
                
                val_sub_acc = accuracy_score(val_targets_sub, val_preds_sub)
                val_group_acc = accuracy_score(val_targets_group, val_preds_group)
                
                print(f" Ep {epoch+1}/{EPOCHS}: Loss {epoch_loss:.4f} | Train S/G: {train_sub_acc:.3f}/{train_group_acc:.3f} | Val S/G: {val_sub_acc:.3f}/{val_group_acc:.3f}")

            # Use Final Epoch results for Fold Metrics
            all_preds = val_preds_sub
            all_targets = val_targets_sub
            all_group_preds = val_preds_group
            all_group_targets = val_targets_group
            
            acc = accuracy_score(all_targets, all_preds)
            f1_m = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            f1_w = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            
            cm = confusion_matrix(all_targets, all_preds, labels=list(range(10))) 
            class_accs = np.nan_to_num(cm.diagonal() / cm.sum(axis=1)).tolist()
            
            cm_group = confusion_matrix(all_group_targets, all_group_preds, labels=list(range(6)))
            group_accs = np.nan_to_num(cm_group.diagonal() / cm_group.sum(axis=1)).tolist()
            
            results['accuracy'].append(acc)
            results['f1_macro'].append(f1_m)
            results['f1_weighted'].append(f1_w)
            
            for i, v in enumerate(class_accs): results['per_class_accuracy'][i].append(v)
            for i, v in enumerate(group_accs): results['per_group_accuracy'][i].append(v)
            
            print(f"Acc: {acc:.4f}")

            # Save Model (Last Fold of Last Repeat effectively becomes 'the' model if valid, 
            # though ideally we pick best val acc. For simplicity matching old script, save last.)
            if repeat == N_REPEATS - 1 and fold == K_FOLDS - 1:
                torch.save(model.state_dict(), RESULTS_DIR / 'mnist_model_final.pth')
                print(f"Saved final model to {RESULTS_DIR / 'mnist_model_final.pth'}")

    # Summary
    json_results = {
        'accuracy': results['accuracy'], 
        'f1_macro': results['f1_macro'],
        'per_class_accuracy': results['per_class_accuracy'],
        'per_group_accuracy': results['per_group_accuracy'],
        'mean_accuracy': float(np.mean(results['accuracy'])),
        'std_accuracy': float(np.std(results['accuracy']))
    }
    
    with open(RESULTS_DIR / 'results_mnist.json', 'w') as f:
        json.dump(json_results, f, indent=4)
        
    print(f"\nSaved results to {RESULTS_DIR / 'results_mnist.json'}")
    print(f"Mean Accuracy: {json_results['mean_accuracy']:.4f}")

if __name__ == '__main__':
    train_cv()
