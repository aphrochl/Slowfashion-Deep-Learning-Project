import re
from collections import Counter

INPUT_FILE = "valid_images.txt"

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("File not found.")
        return

    # Regex to find the anchor "mens-fashion" or "womens-fashion"
    # and capture the parts after it.
    # Pattern: .../{brand}-{gender}-fashion-{category}-{subcategory}-{rest}
    # Note: URL filenames are url-encoded or just hyphens. 
    # Example: .../acne-studios-womens-fashion-clothing-tops-t-shirts-m-front.png
    
    # We look for "-womens-fashion-" or "-mens-fashion-"
    
    categories = Counter()
    subcategories = Counter()
    
    for url in urls:
        filename = url.split('/')[-1]
        
        # Check gender anchor
        if "-womens-fashion-" in filename:
            anchor = "-womens-fashion-"
        elif "-mens-fashion-" in filename:
            anchor = "-mens-fashion-"
        else:
            continue # specific logic for unknown or other patterns?
            
        parts = filename.split(anchor)
        if len(parts) < 2:
            continue
            
        # parts[0] is brand (roughly)
        # parts[1] is category-subcategory-size...
        
        suffix_parts = parts[1].split('-')
        
        if len(suffix_parts) >= 1:
            cat = suffix_parts[0]
            categories[cat] += 1
            
            if len(suffix_parts) >= 2:
                # specific handling for complex subcats could be needed, but let's take next token
                sub = suffix_parts[1]
                subcategories[f"{cat} -> {sub}"] += 1

    print("\n--- Found Categories ---")
    for cat, count in categories.most_common():
        print(f"{cat}: {count}")
        
    print("\n--- Found Subcategories (Top 20) ---")
    for sub, count in subcategories.most_common(20):
        print(f"{sub}: {count}")

if __name__ == "__main__":
    main()
