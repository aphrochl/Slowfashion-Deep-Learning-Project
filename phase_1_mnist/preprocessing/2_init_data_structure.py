import os
import shutil
from pathlib import Path

def setup_structure():
    # Base paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    data_dir = base_dir / 'data'
    
    print(f"Script dir: {script_dir}")
    print(f"Base dir: {base_dir}")

    # Define directories to create
    dirs = [
        data_dir / 'raw',
        data_dir / 'processed',
        data_dir / 'train',
        data_dir / 'val',
        data_dir / 'test'
    ]
    
    # Create directories
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created/Verified directory: {d}")
        
    # Move raw .gz files from mnist_dataset to data/raw
    source_dir = base_dir / 'mnist_dataset'
    target_dir = data_dir / 'raw'

    
    # Check if source exists
    if not source_dir.exists():
        print(f"Warning: Source directory {source_dir} does not exist.")
        return

    # Extensions to look for
    extensions = ['.gz']
    
    files_moved = 0
    for file_path in source_dir.glob('*'):
        if file_path.suffix in extensions:
            target_path = target_dir / file_path.name
            if not target_path.exists():
                shutil.copy2(file_path, target_path) # Using copy to be safe, can switch to move later
                print(f"Copied {file_path.name} to {target_dir}")
                files_moved += 1
            else:
                print(f"Skipped {file_path.name} (already exists in target)")
                
    print(f"Setup complete. {files_moved} files copied to data/raw.")

if __name__ == "__main__":
    setup_structure()
