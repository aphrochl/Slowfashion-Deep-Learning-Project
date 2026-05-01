import gzip
import shutil
import os

def unzip_gz_files(root_dir):
    print(f"Searching for .gz files in {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
                output_path = file_path[:-3] # Remove .gz
                
                print(f"Unzipping {file_path} to {output_path}")
                
                try:
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Successfully unzipped {file_path}")
                except Exception as e:
                    print(f"Error unzipping {file_path}: {e}")

if __name__ == "__main__":
    # Assuming script is run from project root or helpful location
    project_root = os.getcwd()
    unzip_gz_files(project_root)
