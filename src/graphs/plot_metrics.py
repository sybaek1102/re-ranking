import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# [Step 0] Argument parsing and path setup
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot training metrics from a CSV log file.")
parser.add_argument(
    "--csv", type=str, required=True,
    help="Name of the CSV file located in the logs directory (e.g., re-ranking_mlp.csv)"
)
args = parser.parse_args()

# Resolve directories relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Logs are stored under ../../data/output/logs
LOGS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/output/logs"))
# Results graphs directory: ../../results/graphs
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../results/graphs"))

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

csv_path = os.path.join(LOGS_DIR, args.csv)
if not os.path.exists(csv_path):
    print(f"[Error] CSV file not found: {csv_path}")
    exit(1)

# Load data
df_raw = pd.read_csv(csv_path)

# Check if 'fold' column exists
has_folds = 'fold' in df_raw.columns

if has_folds:
    print(f">>> Fold column detected. Computing mean ± std across {df_raw['fold'].nunique()} folds...")
    # Group by epoch and compute mean and std across all folds
    df_mean = df_raw.groupby('epoch').mean().reset_index()
    df_std = df_raw.groupby('epoch').std().reset_index()
    # Drop the 'fold' column
    df_mean = df_mean.drop(columns=['fold'], errors='ignore')
    df_std = df_std.drop(columns=['fold'], errors='ignore')
    print(f">>> Mean data shape: {df_mean.shape}")
    print(f">>> Std data shape: {df_std.shape}")
    df = df_mean
else:
    print(">>> No fold column detected. Using data as-is.")
    df = df_raw
    df_std = None

# Derive output PNG paths (same base name as CSV)
base_name = os.path.splitext(args.csv)[0]
png_path = os.path.join(RESULTS_DIR, f"{base_name}.png")
png_path_shadow = os.path.join(RESULTS_DIR, f"{base_name}_shadow.png") if has_folds else None

# -----------------------------------------------------------------------------
# [Helper Functions]
# -----------------------------------------------------------------------------
def set_style(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

def plot_simple(df):
    """Plot without shadow (simple average lines)"""
    plt.figure(figsize=(24, 12))
    plt.rcParams.update({"font.size": 12})
    
    # Top row (Loss, Accuracy, AUC)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tomato', linestyle='--', linewidth=2)
    set_style(ax1, 'Loss (BCE)', 'Loss')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', color='royalblue', linewidth=2)
    ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', color='tomato', linestyle='--', linewidth=2)
    set_style(ax2, 'Accuracy', 'Accuracy')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df['epoch'], df['train_auc'], label='Train AUC', color='royalblue', linewidth=2)
    ax3.plot(df['epoch'], df['val_auc'], label='Val AUC', color='tomato', linestyle='--', linewidth=2)
    set_style(ax3, 'AUC', 'AUC')
    
    # Bottom row (Precision & Recall)
    ax4 = plt.subplot(2, 4, 5)
    ax4.plot(df['epoch'], df['train_prec0'], label='Train Prec 0', color='royalblue', linewidth=2)
    ax4.plot(df['epoch'], df['val_prec0'], label='Val Prec 0', color='tomato', linestyle='--', linewidth=2)
    set_style(ax4, 'Precision - Class 0', 'Precision')
    
    ax5 = plt.subplot(2, 4, 6)
    ax5.plot(df['epoch'], df['train_prec1'], label='Train Prec 1', color='royalblue', linewidth=2)
    ax5.plot(df['epoch'], df['val_prec1'], label='Val Prec 1', color='tomato', linestyle='--', linewidth=2)
    set_style(ax5, 'Precision - Class 1', 'Precision')
    
    ax6 = plt.subplot(2, 4, 7)
    ax6.plot(df['epoch'], df['train_rec0'], label='Train Rec 0', color='royalblue', linewidth=2)
    ax6.plot(df['epoch'], df['val_rec0'], label='Val Rec 0', color='tomato', linestyle='--', linewidth=2)
    set_style(ax6, 'Recall - Class 0', 'Recall')
    
    ax7 = plt.subplot(2, 4, 8)
    ax7.plot(df['epoch'], df['train_rec1'], label='Train Rec 1', color='royalblue', linewidth=2)
    ax7.plot(df['epoch'], df['val_rec1'], label='Val Rec 1', color='tomato', linestyle='--', linewidth=2)
    set_style(ax7, 'Recall - Class 1', 'Recall')
    
    plt.tight_layout()
    return plt.gcf()

def plot_with_shadow(df, df_std):
    """Plot with shadow (mean ± std shaded area)"""
    plt.figure(figsize=(24, 12))
    plt.rcParams.update({"font.size": 12})
    
    def plot_with_std(ax, epochs, train_mean, val_mean, train_std, val_std, label_prefix=''):
        # Plot mean lines
        ax.plot(epochs, train_mean, label=f'Train {label_prefix}', color='royalblue', linewidth=2)
        ax.plot(epochs, val_mean, label=f'Val {label_prefix}', color='tomato', linestyle='--', linewidth=2)
        
        # Add shaded area for std
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                        color='royalblue', alpha=0.2)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, 
                        color='tomato', alpha=0.2)
    
    # Top row
    ax1 = plt.subplot(2, 3, 1)
    plot_with_std(ax1, df['epoch'], df['train_loss'], df['val_loss'],
                  df_std['train_loss'], df_std['val_loss'], 'Loss')
    set_style(ax1, 'Loss (BCE)', 'Loss')
    
    ax2 = plt.subplot(2, 3, 2)
    plot_with_std(ax2, df['epoch'], df['train_acc'], df['val_acc'],
                  df_std['train_acc'], df_std['val_acc'], 'Acc')
    set_style(ax2, 'Accuracy', 'Accuracy')
    
    ax3 = plt.subplot(2, 3, 3)
    plot_with_std(ax3, df['epoch'], df['train_auc'], df['val_auc'],
                  df_std['train_auc'], df_std['val_auc'], 'AUC')
    set_style(ax3, 'AUC', 'AUC')
    
    # Bottom row
    ax4 = plt.subplot(2, 4, 5)
    plot_with_std(ax4, df['epoch'], df['train_prec0'], df['val_prec0'],
                  df_std['train_prec0'], df_std['val_prec0'], 'Prec 0')
    set_style(ax4, 'Precision - Class 0', 'Precision')
    
    ax5 = plt.subplot(2, 4, 6)
    plot_with_std(ax5, df['epoch'], df['train_prec1'], df['val_prec1'],
                  df_std['train_prec1'], df_std['val_prec1'], 'Prec 1')
    set_style(ax5, 'Precision - Class 1', 'Precision')
    
    ax6 = plt.subplot(2, 4, 7)
    plot_with_std(ax6, df['epoch'], df['train_rec0'], df['val_rec0'],
                  df_std['train_rec0'], df_std['val_rec0'], 'Rec 0')
    set_style(ax6, 'Recall - Class 0', 'Recall')
    
    ax7 = plt.subplot(2, 4, 8)
    plot_with_std(ax7, df['epoch'], df['train_rec1'], df['val_rec1'],
                  df_std['train_rec1'], df_std['val_rec1'], 'Rec 1')
    set_style(ax7, 'Recall - Class 1', 'Recall')
    
    plt.tight_layout()
    return plt.gcf()

# -----------------------------------------------------------------------------
# [Step 1] Generate Plots
# -----------------------------------------------------------------------------
# Always generate simple plot
print(f"\n>>> Generating simple plot...")
fig_simple = plot_simple(df)
fig_simple.savefig(png_path, dpi=300)
plt.close(fig_simple)
print(f">>> Plot saved to: {png_path}")

# Generate shadow plot if fold data exists
if has_folds:
    print(f"\n>>> Generating shadow plot (with std)...")
    fig_shadow = plot_with_shadow(df, df_std)
    fig_shadow.savefig(png_path_shadow, dpi=300)
    plt.close(fig_shadow)
    print(f">>> Shadow plot saved to: {png_path_shadow}")

print("=" * 50)
print(">>> All plots generated successfully!")
print("=" * 50)
