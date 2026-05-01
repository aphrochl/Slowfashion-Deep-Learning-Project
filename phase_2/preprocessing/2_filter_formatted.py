INPUT_ORIGINAL = "slowfashion_all_images.txt"
INPUT_VALID = "valid_images.txt"
OUTPUT_FILE = "slowfashion_valid_images_formatted.txt"

def main():
    print(f"Loading valid URLs from {INPUT_VALID}...")
    try:
        with open(INPUT_VALID, "r", encoding="utf-8") as f:
            valid_urls = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Error: {INPUT_VALID} not found. Please run check_links.py first.")
        return

    print(f"Loaded {len(valid_urls)} valid URLs.")
    
    print(f"Processing {INPUT_ORIGINAL}...")
    with open(INPUT_ORIGINAL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        kept_lines = 0
        total_lines = 0
        
        for line in f_in:
            total_lines += 1
            stripped = line.strip()
            
            # If it's empty, decide whether to keep. Usually keep to preserve spacing.
            if not stripped:
                f_out.write(line)
                continue
                
            # If it's a structural header (doesn't look like a URL)
            if not stripped.startswith("http"):
                f_out.write(line)
                continue
            
            # It is a URL. Check if valid.
            if stripped in valid_urls:
                f_out.write(line)
                kept_lines += 1
            # Else: discard invalid URL
            
    print(f"Finished. Scanned {total_lines} lines.")
    print(f"Kept {kept_lines} valid image links (plus headers/spacing).")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
