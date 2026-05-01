import requests
import concurrent.futures
import time

INPUT_FILE = "slowfashion_all_images.txt"
OUTPUT_FILE = "valid_images.txt"
MAX_WORKERS = 20  # Adjust based on network capabilities

def validate_url(url):
    url = url.strip()
    if not url.startswith("http"):
        return None  # Skip non-URL lines like "FRONT:" or "BACK:"
    
    try:
        # Use HEAD request first for speed
        r = requests.head(url, timeout=5)
        
        # If HEAD fails or gives distinct 404/403, we might stop there.
        # But sometimes HEAD is blocked while GET works. 
        # Given the user's input, 404 is the main "bad" indicator.
        
        if r.status_code == 200:
             content_type = r.headers.get("Content-Type", "").lower()
             if "image" in content_type:
                 return url
        
        # Fallback to GET if HEAD was weird but not strictly failure, 
        # or if we want to be absolutely sure (e.g. some servers return 405 Method Not Allowed for HEAD)
        # For this specific site, the user showed a 404 for bad links, which HEAD should catch.
        # However, let's just stick to the plan: if HEAD != 200, it's likely bad or requires GET.
        # Let's try GET if status is not 404/403 just to be safe, or just rely on HEAD if it's 200.
        
        if r.status_code == 404:
            return None
        
        if r.status_code != 200:
             # Try GET just in case
             r = requests.get(url, stream=True, timeout=5)
             if r.status_code == 200:
                 content_type = r.headers.get("Content-Type", "").lower()
                 if "image" in content_type:
                     return url
                     
    except requests.RequestException:
        pass
        
    return None

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(lines)} lines. Checking URLs...")
    
    valid_urls = []
    processed = 0
    total_checks = 0
    
    # Filter out obvious non-urls first to save thread overhead
    urls_to_check = [u for u in lines if u.startswith("http")]
    print(f"Identified {len(urls_to_check)} potential URLs to check.")

    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(validate_url, url): url for url in urls_to_check}
        
        for future in concurrent.futures.as_completed(future_to_url):
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(urls_to_check)}...")
                
            result = future.result()
            if result:
                valid_urls.append(result)

    duration = time.time() - start_time
    print(f"Finished in {duration:.2f} seconds.")
    print(f"Found {len(valid_urls)} valid images out of {len(urls_to_check)} checked.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for url in valid_urls:
            f.write(url + "\n")
            
    print(f"Saved valid URLs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
