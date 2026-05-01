import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path

# Categories Definition
SUB_CATEGORIES = [
    'Dresses', 'High Heels', 'Shoulder Bags', 'Skirts', 
    'Tote Bags', 'Clutches', 'Outerwear', 'Boots', 'Flats'
]
# Added 'Other' as the 10th subcategory (index 9)

GROUPS = ['Bags', 'Shoes', 'Clothing', 'Jewellery', 'Accessories', 'Other']

# Mappings
SUB_TO_IDX = {name: i for i, name in enumerate(SUB_CATEGORIES)}
GROUP_TO_IDX = {name: i for i, name in enumerate(GROUPS)}

class SlowFashionCsvDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, grayscale=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            grayscale (bool): If True, open images in Grayscale (L) mode.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.SUB_TO_IDX = SUB_TO_IDX 
        self.GROUP_TO_IDX = GROUP_TO_IDX
        
        # Load Data
        all_data = pd.read_csv(csv_file)
        
        self.data = all_data.reset_index(drop=True)
        
        print(f"Dataset initialized from {csv_file}")
        print(f" Total Rows: {len(self.data)} (Including all Groups/Subcategories)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        
        # Image Loading
        img_path = self.root_dir / row['local_path']
        
        # --- Fallback for Encoding Issues (UUID Lookup) ---
        if not img_path.exists():
            # filename is like "uuid_rest.jpg". 
            # We assume the first 36 chars are a UUID which is safe.
            fname = img_path.name
            parent = img_path.parent
            if len(fname) > 36:
                uuid_prefix = fname[:36]
                # Search in the folder for a file starting with this uuid
                if parent.exists():
                    for f in os.listdir(parent):
                        if f.startswith(uuid_prefix):
                            img_path = parent / f
                            # print(f"Recovered: {fname} -> {f}")
                            break
        
        try:
            if self.grayscale:
                image = Image.open(img_path).convert('L')
            else:
                image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if self.grayscale:
                image = Image.new('L', (28, 28))
            else:
                image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # Labels
        sub_name = row['label']
        group_name = row['group']
        
        # Subcategory Mapping
        if sub_name in SUB_TO_IDX:
            sub_label = SUB_TO_IDX[sub_name]
        else:
            sub_label = 9 # 'Other' class

        # Group Mapping
        if group_name in GROUP_TO_IDX:
            group_label = GROUP_TO_IDX[group_name]
        else:
            group_label = GROUP_TO_IDX['Other']

        return image, group_label, sub_label
