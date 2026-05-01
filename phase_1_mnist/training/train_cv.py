import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import time

# ----------------- CONFIG -----------------
BATCH_SIZE = 64
EPOCHS = 3 
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_REPEATS = 10  
K_FOLDS = 10

# ----------------- MODEL -----------------
class DualOutputCNN(nn.Module):
    def __init__(self, num_groups=3, num_sub=9):
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
    print(f"Starting Cross-Validation on {DEVICE}")
    
    # Load TRAIN Data (90% of total)
    base_dir = Path(__file__).resolve().parent.parent # Slowfashion/
    
    # Portable relative path (Works if 'data' folder is inside project root)
    data_path = base_dir / 'data' / 'train' / 'data.pt'
    
    if not data_path.exists():
        print(f"Training data not found at {data_path}!")
        print("Ensure you have unzipped 'slowfashion_data.zip' into the project root.")
        return
        
    print(f"Loading training data from {data_path} for Cross-Validation...")
    data_dict = torch.load(data_path)
    X_all = data_dict['data']
    y_sub_all = data_dict['targets_sub']
    y_group_all = data_dict['targets_group']
    
    print(f"Total CV Data: {len(X_all)} samples")
    
    y_stratify = y_sub_all.numpy()
    
    results = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': []
    }
    
    start_time = time.time()
    
    for repeat in range(N_REPEATS):
        print(f"\n--- Repeat {repeat+1}/{N_REPEATS} ---")
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42 + repeat)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_stratify)):
            print(f"  Fold {fold+1}/{K_FOLDS}", end='... ')
            
            # Prepare Data
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_group_train, y_group_val = y_group_all[train_idx], y_group_all[val_idx]
            y_sub_train, y_sub_val = y_sub_all[train_idx], y_sub_all[val_idx]
            
            train_ds = TensorDataset(X_train, y_group_train, y_sub_train)
            val_ds = TensorDataset(X_val, y_group_val, y_sub_val)
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            # Init Model
            model = DualOutputCNN().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()
            
            # Training Loop
            for epoch in range(EPOCHS):
                model.train()
                for X_b, y_g_b, y_s_b in train_loader:
                    X_b, y_g_b, y_s_b = X_b.to(DEVICE), y_g_b.to(DEVICE), y_s_b.to(DEVICE)
                    optimizer.zero_grad()
                    out_g, out_s = model(X_b)
                    loss = criterion(out_g, y_g_b) + criterion(out_s, y_s_b)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation
            model.eval()
            all_preds = []
            all_targets = []
            all_group_preds = []
            all_group_targets = []
            
            with torch.no_grad():
                for X_b, y_g_b, y_s_b in val_loader:
                    X_b = X_b.to(DEVICE)
                    out_g, out_s = model(X_b)
                    
                    _, preds = torch.max(out_s, 1)
                    _, group_preds = torch.max(out_g, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_s_b.cpu().numpy())
                    
                    all_group_preds.extend(group_preds.cpu().numpy())
                    all_group_targets.extend(y_g_b.cpu().numpy())
            
            # Metrics (Subcategory)
            acc = accuracy_score(all_targets, all_preds)
            f1_m = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            f1_w = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            
            # Metrics (Group)
            acc_group = accuracy_score(all_group_targets, all_group_preds)
            
            # --- Per-Class Accuracy (Subcategory) ---
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_targets, all_preds, labels=list(range(9))) 
            class_accs = cm.diagonal() / cm.sum(axis=1)
            class_accs = np.nan_to_num(class_accs).tolist()
            
            # --- Per-Class Accuracy (Group) ---
            cm_group = confusion_matrix(all_group_targets, all_group_preds, labels=[0, 1, 2]) # 0:Bags, 1:Shoes, 2:Clothing
            group_accs = cm_group.diagonal() / cm_group.sum(axis=1)
            group_accs = np.nan_to_num(group_accs).tolist()

            # Store results
            results['accuracy'].append(acc)
            results['f1_macro'].append(f1_m)
            results['f1_weighted'].append(f1_w)
            
            # Init per-class storage if first fold
            if 'per_class_accuracy' not in results:
                results['per_class_accuracy'] = {i: [] for i in range(9)}
                results['per_group_accuracy'] = {i: [] for i in range(3)} # 3 Groups
            
            for cls_idx, cls_acc in enumerate(class_accs):
                results['per_class_accuracy'][cls_idx].append(cls_acc)
                
            for grp_idx, grp_acc in enumerate(group_accs):
                results['per_group_accuracy'][grp_idx].append(grp_acc)
            
            print(f"Sub Acc: {acc:.4f} | Group Acc: {acc_group:.4f}")
            
            # Save final model (last fold, last repeat)
            if repeat == N_REPEATS - 1 and fold == K_FOLDS - 1:
                torch.save(model.state_dict(), 'slowfashion_model_final.pth')
                print("Saved final model weights to 'slowfashion_model_final.pth'")

    total_time = time.time() - start_time
    print(f"\nCompleted {N_REPEATS}x{K_FOLDS}-Fold CV in {total_time:.1f}s")
    
    # Save Detailed Results to JSON for Visualization
    import json
    json_results = {
        'accuracy': results['accuracy'], 
        'f1_macro': results['f1_macro'],
        'per_class_accuracy': results['per_class_accuracy'],
        'per_group_accuracy': results['per_group_accuracy'], # New!
        'mean_accuracy': float(np.mean(results['accuracy'])),
        'std_accuracy': float(np.std(results['accuracy']))
    }
    
    with open('cv_results_for_viz.json', 'w') as f:
        json.dump(json_results, f, indent=4)
        
    print("Saved detailed metrics to 'cv_results_for_viz.json'")
    print(f"Mean Subcategory Accuracy: {json_results['mean_accuracy']:.4f}")

if __name__ == "__main__":
    train_cv()
