import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# ----------------- CONFIG -----------------
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'slowfashion_model_final.pth'

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

def test_model():
    print(f"Starting Testing on {DEVICE}")
    
    # 1. Load Test Data
    base_dir = Path(__file__).resolve().parent.parent # Slowfashion/
    
    # Portable/Server path check
    data_path = base_dir / 'data' / 'test' / 'data.pt'
    
    if not data_path.exists():
        print(f"Test data not found at {data_path}!")
        print("Ensure 'data' folder is in the project root (unzip slowfashion_data.zip there).")
        return
        
    print(f"Loading TEST data from {data_path}...")
    data_dict = torch.load(data_path)
    X_test = data_dict['data']
    y_sub_test = data_dict['targets_sub']
    y_group_test = data_dict['targets_group']
    
    test_ds = TensorDataset(X_test, y_group_test, y_sub_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Total Test Data: {len(X_test)} samples")
    
    # 2. Load Model
    model = DualOutputCNN().to(DEVICE)
    model_name = 'slowfashion_model_final.pth'
    
    # Check current dir, parent dir, or training dir relative to script
    candidate_paths = [
        Path(model_name),                                      # current dir
        base_dir / model_name,                                 # project root
        base_dir / 'training' / model_name                     # training dir
    ]
    
    model_file = None
    for p in candidate_paths:
        if p.exists():
            model_file = p
            break
            
    if model_file is None:
        print(f"Error: Model weights '{model_name}' not found.")
        print("Did you run 'python training/train_cv.py' first?")
        return

    print(f"Loading weights from {model_file}...")
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()
    
    # 3. Evaluate
    all_preds = []
    all_targets = []
    all_group_preds = []
    all_group_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for X_b, y_g_b, y_s_b in test_loader:
            X_b = X_b.to(DEVICE)
            out_g, out_s = model(X_b)
            
            _, preds = torch.max(out_s, 1)
            _, group_preds = torch.max(out_g, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_s_b.cpu().numpy())
            
            # Capture Group results
            all_group_preds.extend(group_preds.cpu().numpy())
            all_group_targets.extend(y_g_b.cpu().numpy())
            
    # 4. Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1_m = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_w = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Group Metrics
    acc_group = accuracy_score(all_group_targets, all_group_preds)
    
    print("\n" + "="*30)
    print("       FINAL TEST RESULTS       ")
    print("="*30)
    print(f"Subcategory Accuracy: {acc:.4f}")
    print(f"Group Accuracy:       {acc_group:.4f}")
    print(f"F1 (Macro):           {f1_m:.4f}")
    print(f"F1 (Weighted):        {f1_w:.4f}")
    print("="*30)
    print("These are the results you should report as 'Test Set Performance'.")

if __name__ == "__main__":
    test_model()
