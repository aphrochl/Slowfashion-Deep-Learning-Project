import csv
import os
import requests
import concurrent.futures
import time

INPUT_CSV = "labeled_dataset.csv"
OUTPUT_CSV = "local_dataset.csv"
DATA_DIR = "data/images"
MAX_WORKERS = 20

def download_image(row):
    # row: [image_url, group, label, filename]
    url = row[0]
    group = row[1]
    label = row[2]
    filename = row[3]
    
    # Sanitize inputs
    safe_group = "".join([c for c in group if c.isalnum() or c in (' ', '-', '_')]).strip()
    safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '-', '_')]).strip()
    
    # Extract Product ID from URL to ensure uniqueness
    # URL format: .../products/{uuid}/{filename}
    try:
        product_id = url.split('/')[-2]
    except:
        product_id = "unknown"
        
    safe_filename = f"{product_id}_{filename}"
    safe_filename = "".join([c for c in safe_filename if c.isalnum() or c in (' ', '-', '_', '.')]).strip()
    
    # Determine save path
    # Structure: data/images/{group}/{label}/{filename}
    save_dir = os.path.join(DATA_DIR, safe_group, safe_label)
    os.makedirs(save_dir, exist_ok=True)
    
    local_path = os.path.join(save_dir, safe_filename)
    
    # If file exists and size > 0, skip download (resume capability)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return row + [local_path]

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return row + [local_path]
    except Exception as e:
        # print(f"Failed to download {url}: {e}")
        pass
        
    return None

def main():
    print(f"Reading {INPUT_CSV}...")
    dataset = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        dataset = list(reader)

    print(f"Found {len(dataset)} images to download. Saving to {DATA_DIR}...")
    
    start_time = time.time()
    
    succeeded = []
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures to rows
        futures = {executor.submit(download_image, row): row for row in dataset}
        
        completed_count = 0
        total = len(dataset)
        
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            if completed_count % 100 == 0:
                print(f"Progress: {completed_count}/{total}...")
                
            res = future.result()
            if res:
                succeeded.append(res)
            else:
                failed += 1
                
    duration = time.time() - start_time
    print(f"Finished in {duration:.2f} seconds.")
    print(f"Successfully downloaded: {len(succeeded)}")
    print(f"Failed: {failed}")
    
    # Save local dataset
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_url", "group", "label", "filename", "local_path"])
        writer.writerows(succeeded)
        
    print(f"Saved local dataset mapping to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
