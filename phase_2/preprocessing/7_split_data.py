
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Config
INPUT_CSV = 'flat_dataset.csv'
INPUT_DIR = 'data/all_images'
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'

# Ensure output dirs exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# Check for existence of files
# Only keep rows where file exists
valid_rows = []
for idx, row in df.iterrows():
    filename = os.path.basename(row['local_path'])
    src_path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(src_path):
        valid_rows.append(row)
    else:
        print(f"Skipping missing file: {filename}")

df = pd.DataFrame(valid_rows)
print(f"Valid entries: {len(df)}")

# Stratified Split
# Stratify by 'label' to ensure small classes (like Tote Bags) are distributed
print("Performing stratified split (90% Train, 10% Test)...")
train_df, test_df = train_test_split(
    df, 
    test_size=0.10, 
    stratify=df['label'], 
    random_state=42
)

def move_files(dataframe, target_dir):
    new_paths = []
    count = 0
    for idx, row in dataframe.iterrows():
        filename = os.path.basename(row['local_path'])
        src_path = os.path.join(INPUT_DIR, filename)
        dst_path = os.path.join(target_dir, filename)
        
        try:
            shutil.move(src_path, dst_path)
            new_paths.append(dst_path.replace('\\', '/'))
            count += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")
            new_paths.append(None) # Mark as failed
            
    return new_paths

# Move Train
print(f"Moving {len(train_df)} files to {TRAIN_DIR}...")
train_paths = move_files(train_df, TRAIN_DIR)
train_df['local_path'] = train_paths

# Move Test
print(f"Moving {len(test_df)} files to {TEST_DIR}...")
test_paths = move_files(test_df, TEST_DIR)
test_df['local_path'] = test_paths

# Save CSVs
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("Split complete.")
print(f"Train: {len(train_df)} images -> {TRAIN_CSV}")
print(f"Test: {len(test_df)} images -> {TEST_CSV}")

# Optional: Clean up empty source dir
# try:
#     os.rmdir(INPUT_DIR)
# except:
#     pass
