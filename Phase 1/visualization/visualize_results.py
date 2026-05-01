
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

# Subcategories Mapping
SUBCATEGORIES = {
    "0": "Dresses",
    "1": "High Heels",
    "2": "Shoulder Bags",
    "3": "Skirts",
    "4": "Tote Bags",
    "5": "Clutches",
    "6": "Outerwear",
    "7": "Boots",
    "8": "Flats"
}

# Groups Mapping
GROUPS = {0: 'Bags', 1: 'Shoes', 2: 'Clothing'}

def load_results(filename='cv_results_custom.json'):
    # Look in current dir or parent dir or training dir
    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent.parent / filename,
        Path(__file__).resolve().parent.parent / 'training' / filename
    ]
    
    for path in candidates:
        if path.exists():
            print(f"Loading results from: {path}")
            with open(path, 'r') as f:
                return json.load(f)
    
    print(f"Error: {filename} not found.")
    print("Checked locations:")
    for c in candidates: print(f" - {c}")
    return None

def plot_boxplot(results):
    print("Generating Box Plot...")
    
    per_class_acc = results['per_class_accuracy']
    
    # Prepare data for plotting
    data = []
    labels = []
    
    # Sort by key to ensure order
    sorted_keys = sorted(per_class_acc.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        values = per_class_acc[key]
        # Only plot if we have data (handle sparse custom dataset)
        if len(values) > 0 and np.mean(values) > 0.0:
            label = SUBCATEGORIES.get(key, f"Class {key}")
            data.append(values)
            labels.append(label)
        
    global_mean = results['mean_accuracy']

    if not data:
        print("No valid per-class data found to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="#e6f2ff", color="#003366"),
                medianprops=dict(color="black"))

    plt.axhline(y=global_mean, color='r', linestyle='--', label=f'Global Mean ({global_mean:.4f})')
    plt.title('Distribution of Classification Accuracy per Category')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Fashion Categories')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_name = 'accuracy_boxplot.png'
    plt.savefig(output_name, dpi=300)
    print(f"Saved {output_name}")

def plot_group_boxplot(results):
    print("Generating Group Box Plot...")
    
    per_group_acc = results.get('per_group_accuracy', {})
    if not per_group_acc:
        print("No group data found, skipping group plot.")
        return

    data = []
    labels = []
    
    sorted_keys = sorted(per_group_acc.keys(), key=lambda x: int(x))
    
    for key in sorted_keys:
        values = per_group_acc[key]
        if len(values) > 0 and np.mean(values) > 0.0:
            label = GROUPS.get(int(key), f"Group {key}")
            data.append(values)
            labels.append(label)
            
    if not data:
        print("No valid group data found to plot.")
        return
        
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="#c2e0c6", color="#004d00"), 
                medianprops=dict(color="black"))

    plt.title('Distribution of Classification Accuracy per Broad Category')
    plt.ylabel('Accuracy Score')
    plt.xlabel('Broad Categories')
    plt.ylim(0.0, 1.05) 
    plt.tight_layout()
    
    output_name = 'group_accuracy_boxplot.png'
    plt.savefig(output_name, dpi=300)
    print(f"Saved {output_name}")

def plot_bayesian(results, baseline=0.91):
    print("Generating Bayesian Posterior Plot...")
    
    acc_model = results['mean_accuracy']
    std_dev = results['std_accuracy']
    
    if std_dev == 0:
        std_dev = 0.001 # Avoid division by zero if single run/perfect stability
    
    diff = acc_model - baseline
    x = np.linspace(diff - 4*std_dev, diff + 4*std_dev, 1000)
    
    pdf = stats.norm.pdf(x, diff, std_dev)

    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf, label='Posterior Distribution of Difference', color='#2ca02c', linewidth=2)
    plt.fill_between(x, pdf, color='#98df8a', alpha=0.4)

    plt.axvline(x=-0.01, color='orange', linestyle='-', alpha=0.7)
    plt.axvline(x=0.01, color='orange', linestyle='-', alpha=0.7, label='ROPE (+/- 1%)')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.title(f'Bayesian Posterior Distribution of Accuracy Difference\n(Model {acc_model*100:.1f}% vs. Baseline {baseline*100:.1f}%)')
    plt.xlabel('Difference in Accuracy (Model - Baseline)')
    plt.yticks([]) 
    plt.legend(loc='upper left')

    plt.text(diff, max(pdf)*0.8, f"Mean Shift: {diff:+.4f}", 
             ha='center', color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_name = 'bayesian_posterior.png'
    plt.savefig(output_name, dpi=300)
    print(f"Saved {output_name}")

if __name__ == "__main__":
    results = load_results()
    if results:
        plot_boxplot(results)
        plot_group_boxplot(results)
        plot_bayesian(results)
        print("Visualization generation complete.")
