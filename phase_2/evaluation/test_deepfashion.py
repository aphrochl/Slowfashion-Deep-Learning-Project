import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from pathlib import Path
from torchvision import transforms, models

# Path Setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from training.dataset import SlowFashionCsvDataset
except ImportError:
    from training.dataset import SlowFashionCsvDataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
TEST_CSV = BASE_DIR / 'data' / 'metadata' / 'test.csv'
MODEL_PATH = BASE_DIR / "deepfashion_model" # Or .pth if renamed

# Labels
SUB_NAMES = [
    'Dresses', 'High Heels', 'Shoulder Bags', 'Skirts', 
    'Tote Bags', 'Clutches', 'Outerwear', 'Boots', 'Flats', 'Other'
]
GROUP_NAMES = ['Bags', 'Shoes', 'Clothing', 'Jewellery', 'Accessories', 'Other']

# --- Model Definition ---
class DualOutputResNet(nn.Module):
    def __init__(self, num_groups=6, num_sub=10):
        super(DualOutputResNet, self).__init__()
        self.backbone = models.resnet18(weights=None) 
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

def test_deepfashion():
    print(f"Starting DeepFashion Testing on {DEVICE}")
    print(f"Metadata: {TEST_CSV}")
    
    # Locate Model
    candidates = [
        BASE_DIR / 'results' / 'deepfashion_model_final.pth',
        BASE_DIR / 'deepfashion_model.pth',
        BASE_DIR / 'deepfashion_model'
    ]
    
    model_file = None
    for c in candidates:
        if c.exists():
            model_file = c
            break
            
    if model_file is None:
        print(f"Error: Model not found. Checked locations:")
        for c in candidates: print(f" - {c}")
        return

    # Transforms (224x224 RGB)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SlowFashionCsvDataset(TEST_CSV, BASE_DIR, transform=transform, grayscale=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(dataset) == 0:
        print("Dataset empty. Exiting.")
        return

    model = DualOutputResNet(num_groups=6, num_sub=10).to(DEVICE)
    
    try:
        state = torch.load(model_file, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {model_file}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    all_preds_sub = []
    all_targets_sub = []
    all_preds_group = []
    all_targets_group = []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, group_labels, sub_labels in loader:
            inputs = inputs.to(DEVICE)
            out_group, out_sub = model(inputs)
            
            _, pred_sub = out_sub.max(1)
            _, pred_group = out_group.max(1)
            
            all_preds_sub.extend(pred_sub.cpu().numpy())
            all_targets_sub.extend(sub_labels.numpy())
            all_preds_group.extend(pred_group.cpu().numpy())
            all_targets_group.extend(group_labels.numpy())

    # Metrics
    acc_sub = accuracy_score(all_targets_sub, all_preds_sub)
    f1_sub = f1_score(all_targets_sub, all_preds_sub, average='macro', zero_division=0)
    acc_group = accuracy_score(all_targets_group, all_preds_group)
    
    print("\n" + "="*40)
    print("       DEEPFASHION MODEL TEST RESULTS       ")
    print("="*40)
    print(f"Subcategory Accuracy:   {acc_sub:.4f}")
    print(f"Subcategory F1 (Macro): {f1_sub:.4f}")
    print(f"Group Accuracy:         {acc_group:.4f}")
    print("-" * 20)
    print(classification_report(all_targets_sub, all_preds_sub, target_names=SUB_NAMES, labels=list(range(10)), zero_division=0))
    print("="*40)

if __name__ == "__main__":
    test_deepfashion()
