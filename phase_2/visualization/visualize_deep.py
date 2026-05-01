import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import json
from pathlib import Path

# ---------------------------------------------------------
# CONSTANTS & SETUP
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

# NEW MAPPINGS
SUB_NAMES = [
    'Dresses', 'High Heels', 'Shoulder Bags', 'Skirts', 
    'Tote Bags', 'Clutches', 'Outerwear', 'Boots', 'Flats', 'Other'
]
GROUP_NAMES = ['Bags', 'Shoes', 'Clothing', 'Jewellery', 'Accessories', 'Other']

SUBCATEGORIES = {str(i): name for i, name in enumerate(SUB_NAMES)}
GROUPS = {i: name for i, name in enumerate(GROUP_NAMES)}

def load_results(filename='results_deep.json'):
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / 'results' / filename
    
    if path.exists():
        print(f"Loading results from: {path}")
        with open(path, 'r') as f:
            return json.load(f)
    
    print(f"Error: {path} not found.")
    return None

def plot_boxplot(results):
    print("Generating DeepFashion Subcategory Box Plot...")
    
    per_class_acc = results['per_class_accuracy']
    
    data = []
    labels = []
    
    sorted_keys = sorted(per_class_acc.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        values = per_class_acc[key]
        if len(values) > 0:
            label = SUBCATEGORIES.get(key, f"Class {key}")
            data.append(values)
            labels.append(label)
        
    global_mean = results['mean_accuracy']

    if not data:
        print("No valid per-class data found.")
        return

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="#ffe6e6", color="#800000"), # Different color for DeepFashion (Reddish)
                medianprops=dict(color="black"))

    plt.axhline(y=global_mean, color='blue', linestyle='--', label=f'Global Mean ({global_mean:.4f})')
    plt.title('DeepFashion Model: Accuracy per Subcategory')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Subcategories')
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_name = 'deep_subcategory_boxplot.png'
    plt.savefig(output_name, dpi=300)
    print(f"Saved {output_name}")

def plot_group_boxplot(results):
    print("Generating DeepFashion Group Box Plot...")
    
    per_group_acc = results.get('per_group_accuracy', {})
    if not per_group_acc:
        print("No group data found.")
        return

    data = []
    labels = []
    
    sorted_keys = sorted(per_group_acc.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        values = per_group_acc[key]
        if len(values) > 0:
            label = GROUPS.get(int(key), f"Group {key}")
            data.append(values)
            labels.append(label)
            
    if not data:
        print("No valid group data found.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="#e6ccff", color="#4b0082"), # Purple-ish
                medianprops=dict(color="black"))

    plt.title('DeepFashion Model: Accuracy per Broad Group')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Broad Groups')
    plt.ylim(0.0, 1.05) 
    plt.tight_layout()
    
    output_name = 'deep_group_boxplot.png'
    plt.savefig(output_name, dpi=300)
    print(f"Saved {output_name}")

def plot_bayesian(results, baseline=0.66):
    print("Generating Bayesian Posterior Plot...")
    
    acc_model = results['mean_accuracy']
    std_dev = results['std_accuracy']
    
    if std_dev == 0:
        std_dev = 0.001
    
    diff = acc_model - baseline
    x = np.linspace(diff - 4*std_dev, diff + 4*std_dev, 1000)
    
    pdf = stats.norm.pdf(x, diff, std_dev)

    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf, label='Posterior Distribution', color='#d62728', linewidth=2) # Red
    plt.fill_between(x, pdf, color='#ff9896', alpha=0.4)

    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.title(f'DeepFashion Model: Bayesian Posterior vs Baseline ({baseline*100:.1f}%)')
    plt.xlabel('Difference in Accuracy (Model - Baseline)')
    plt.yticks([]) 
    plt.legend(loc='upper left')

    plt.text(diff, max(pdf)*0.8, f"Mean Shift: {diff:+.4f}", 
             ha='center', color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_name = 'deep_bayesian_posterior.png'
    plt.savefig(output_name, dpi=300)
    print(f"Saved {output_name}")

if __name__ == "__main__":
    results = load_results('results_deep.json')
    if results:
        plot_boxplot(results)
        plot_group_boxplot(results)
        plot_bayesian(results)
        print("Done.")
