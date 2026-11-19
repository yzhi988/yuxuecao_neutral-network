"""
Training Curves Visualization Tool
Plots training, validation, and test MSE curves from training logs
Style: CNS-level academic publication standard
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_training_log(log_path):
    """
    Parse training log file to extract MSE values
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Dictionary containing epochs and MSE values for train/valid/test sets
    """
    train_data = {'epoch': [], 'mse': []}
    valid_data = {'epoch': [], 'mse': []}
    test_data = {'epoch': [], 'mse': []}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse training data loss (MSE)
            train_match = re.search(r'\[Train\] epoch:(\d+),.*?data loss:([\d.]+)', line)
            if train_match:
                epoch = int(train_match.group(1))
                mse = float(train_match.group(2))
                train_data['epoch'].append(epoch)
                train_data['mse'].append(mse)
            
            # Parse validation MSE
            valid_match = re.search(r'\[Valid\] epoch:(\d+), MSE: ([\d.e-]+)', line)
            if valid_match:
                epoch = int(valid_match.group(1))
                mse = float(valid_match.group(2))
                valid_data['epoch'].append(epoch)
                valid_data['mse'].append(mse)
            
            # Parse test MSE
            test_match = re.search(r'\[Test\] MSE: ([\d.e-]+)', line)
            if test_match:
                mse = float(test_match.group(1))
                # Get the epoch from the previous validation line
                if valid_data['epoch']:
                    epoch = valid_data['epoch'][-1]
                    test_data['epoch'].append(epoch)
                    test_data['mse'].append(mse)
    
    return train_data, valid_data, test_data


def plot_training_curves(train_data, valid_data, test_data, save_path, 
                         title='Training Convergence', dpi=300):
    """
    Plot training curves with CNS-level academic style
    
    Args:
        train_data: Dictionary with 'epoch' and 'mse' keys for training data
        valid_data: Dictionary with 'epoch' and 'mse' keys for validation data
        test_data: Dictionary with 'epoch' and 'mse' keys for test data
        save_path: Path to save the figure
        title: Figure title
        dpi: Resolution for saved figure
    """
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 4,
        'figure.dpi': 100,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create figure with appropriate size (single column: 3.5", double column: 7")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Define color scheme (colorblind-friendly palette)
    colors = {
        'train': '#0173B2',    # Blue
        'valid': '#DE8F05',    # Orange
        'test': '#029E73'      # Green
    }
    
    # Plot curves with improved styling
    if train_data['epoch']:
        ax.plot(train_data['epoch'], train_data['mse'], 
                color=colors['train'], linewidth=2.0, 
                label='Training', alpha=0.9, zorder=3)
    
    if valid_data['epoch']:
        ax.plot(valid_data['epoch'], valid_data['mse'], 
                color=colors['valid'], linewidth=2.0, 
                label='Validation', alpha=0.9, zorder=4)
    
    if test_data['epoch']:
        ax.plot(test_data['epoch'], test_data['mse'], 
                color=colors['test'], linewidth=2.5, 
                marker='o', markersize=6, markevery=1,
                label='Test', alpha=0.9, linestyle='-', zorder=5)
    
    # Set labels and title
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    
    # Set log scale for y-axis if values span multiple orders of magnitude
    if train_data['mse']:
        max_mse = max(train_data['mse'])
        min_mse = min([m for m in train_data['mse'] + valid_data['mse'] + test_data['mse'] if m > 0])
        if max_mse / min_mse > 100:
            ax.set_yscale('log')
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    
    # Legend with better positioning
    legend = ax.legend(loc='best', framealpha=0.95, edgecolor='black', 
                      fancybox=False, shadow=False)
    legend.get_frame().set_linewidth(1.5)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure in multiple formats
    save_path = Path(save_path)
    base_path = save_path.with_suffix('')
    
    # Create parent directory if it doesn't exist
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF (vector format, preferred for publications)
    plt.savefig(f"{base_path}.pdf", format='pdf', dpi=dpi, bbox_inches='tight')
    print(f"Figure saved as: {base_path}.pdf")
    
    # Save as PNG (raster format, for presentations)
    plt.savefig(f"{base_path}.png", format='png', dpi=dpi, bbox_inches='tight')
    print(f"Figure saved as: {base_path}.png")
    
    # Save as SVG (vector format, editable)
    plt.savefig(f"{base_path}.svg", format='svg', dpi=dpi, bbox_inches='tight')
    print(f"Figure saved as: {base_path}.svg")
    
    plt.show()
    plt.close()


def main(log_path, save_path, title='Training Convergence Curves'):
    """
    Main function to parse log and generate plot
    
    Args:
        log_path: Path to training log file
        save_path: Path to save the output figure (without extension)
        title: Title for the plot
    """
    print(f"Reading log file: {log_path}")
    train_data, valid_data, test_data = parse_training_log(log_path)
    
    print(f"\nData extracted:")
    print(f"  Training epochs: {len(train_data['epoch'])}")
    print(f"  Validation epochs: {len(valid_data['epoch'])}")
    print(f"  Test evaluations: {len(test_data['epoch'])}")
    
    if train_data['epoch']:
        print(f"\nTraining MSE range: {min(train_data['mse']):.6f} - {max(train_data['mse']):.6f}")
    if valid_data['epoch']:
        print(f"Validation MSE range: {min(valid_data['mse']):.6e} - {max(valid_data['mse']):.6e}")
    if test_data['epoch']:
        print(f"Test MSE range: {min(test_data['mse']):.6e} - {max(test_data['mse']):.6e}")
    
    print(f"\nGenerating publication-quality figure...")
    plot_training_curves(train_data, valid_data, test_data, save_path, title=title)
    
    print("\nDone!")


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    LOG_PATH = "实验结果/MITfuzalogging.txt"
    SAVE_PATH = "实验结果/training_curves"
    TITLE = "Training Convergence Curves"
    
    # Run the plotting
    main(LOG_PATH, SAVE_PATH, TITLE)

