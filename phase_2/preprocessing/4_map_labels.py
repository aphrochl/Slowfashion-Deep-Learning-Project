import csv
import re

INPUT_FILE = "valid_images.txt"
OUTPUT_CSV = "labeled_dataset.csv"

# Target Categories
c_dresses = "Dresses"
c_high_heels = "High Heels"
c_shoulder_bags = "Shoulder Bags"
c_skirts = "Skirts"
c_tote_bags = "Tote Bags"
c_clutches = "Clutches"
c_outerwear = "Outerwear"
c_boots = "Boots"
c_flats = "Flats"

def get_group(filename):
    # Filename structure: {brand}-{gender}-fashion-{group}-{subcat}-...
    # Find anchor "-fashion-"
    if "-fashion-" in filename:
        parts = filename.split("-fashion-")
        if len(parts) > 1:
            suffix = parts[1]
            tokens = suffix.split('-')
            if tokens:
                raw_group = tokens[0]
                if raw_group == "clothing":
                    return "Clothing"
                elif raw_group == "shoes":
                    return "Shoes"
                elif raw_group == "bags":
                    return "Bags"
                elif raw_group == "jewellery":
                    return "Jewellery"
                elif raw_group == "accessories":
                    return "Accessories"
                
    # Fallback if structure parsing fails, use keywords
    lower = filename.lower()
    if "clothing" in lower or "dress" in lower or "shirt" in lower or "skirt" in lower:
        return "Clothing"
    if "shoes" in lower or "boot" in lower or "heel" in lower or "sneaker" in lower:
        return "Shoes"
    if "bag" in lower or "tote" in lower:
        return "Bags"
    if "jewel" in lower or "necklace" in lower or "ring" in lower or "bracelet" in lower:
        return "Jewellery"
        
    return "Other"

def get_label(filename):
    lower_name = filename.lower()
    
    # Priority matching (specific to general)
    
    # 1. Dresses
    if "dresses" in lower_name:
        return c_dresses
        
    # 2. High Heels
    if "high-heels" in lower_name or ("shoes" in lower_name and "high" in lower_name):
        return c_high_heels
        
    # 3. Shoulder Bags
    if "shoulder-bags" in lower_name or ("bags" in lower_name and "shoulder" in lower_name):
         return c_shoulder_bags
         
    # 4. Skirts
    if "skirts" in lower_name:
        return c_skirts
        
    # 5. Tote Bags
    if "tote" in lower_name:
        return c_tote_bags
        
    # 6. Clutches
    if "clutch" in lower_name:
        return c_clutches
        
    # 7. Outerwear
    if "coats" in lower_name or "jackets" in lower_name or "blazers" in lower_name or "outerwear" in lower_name:
        return c_outerwear
        
    # 8. Boots
    if "boots" in lower_name:
        return c_boots
        
    # 9. Flats
    if "flats" in lower_name:
        return c_flats
    if "sneakers" in lower_name:
        return c_flats 
    if "sandals" in lower_name:
        return c_flats
        
    return "Other"

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("File not found.")
        return

    labeled_data = []
    
    stats_label = {
        c_dresses: 0,
        c_high_heels: 0,
        c_shoulder_bags: 0,
        c_skirts: 0,
        c_tote_bags: 0,
        c_clutches: 0,
        c_outerwear: 0,
        c_boots: 0,
        c_flats: 0,
        "Other": 0
    }
    
    stats_group = {
        "Clothing": 0,
        "Shoes": 0,
        "Bags": 0,
        "Jewellery": 0,
        "Accessories": 0,
        "Other": 0
    }
    
    for url in urls:
        filename = url.split('/')[-1]
        
        label = get_label(filename)
        group = get_group(filename)
        
        # Cross-verification (Optional): Ensure label matches group?
        # heuristic: if label is Dress, group MUST be Clothing.
        if label == c_dresses: group = "Clothing"
        if label == c_skirts: group = "Clothing"
        if label == c_outerwear: group = "Clothing"
        
        if label == c_high_heels: group = "Shoes"
        if label == c_boots: group = "Shoes"
        if label == c_flats: group = "Shoes"
        
        if label == c_shoulder_bags: group = "Bags"
        if label == c_tote_bags: group = "Bags"
        if label == c_clutches: group = "Bags"
        
        stats_label[label] += 1
        stats_group[group] += 1
        
        labeled_data.append([url, group, label, filename])
        
    print("\n--- Group Statistics ---")
    for g, count in stats_group.items():
        print(f"{g}: {count}")

    print("\n--- Label Statistics ---")
    for cat, count in stats_label.items():
        print(f"{cat}: {count}")
        
    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_url", "group", "label", "filename"])
        writer.writerows(labeled_data)
        
    print(f"\nSaved labeled dataset to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
