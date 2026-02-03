import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# [Step 0] Argument parsing and path setup
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot regression training metrics from a CSV log file.")
parser.add_argument(
    "--csv", type=str, required=True,
    help="Name of the CSV file located in the logs directory (e.g., 12_residual_mlp.csv)"
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
def set_style(ax, title, ylabel, ylim=(-0.05, 1.05)):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

def plot_simple(df):
    """Plot without shadow (simple average lines)"""
    plt.figure(figsize=(24, 12))
    plt.rcParams.update({"font.size": 12})
    
    # Calculate best validation values
    best_val_loss = df['val_loss'].min()
    best_val_r2 = df['val_r2'].max()
    best_val_corr = df['val_corr'].max()
    best_val_rmse = df['val_rmse'].min()
    best_val_mae = df['val_mae'].min()
    best_val_acc_like = df['val_acc_like_0.2'].max()
    
    # Top row (Loss, R², Correlation)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tomato', linestyle='--', linewidth=2)
    set_style(ax1, f'Loss (MSE) (best val = {best_val_loss:.4f})', 'Loss', ylim=(0, 1.05))
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df['epoch'], df['val_r2'], label='Val R² Score', color='tomato', linewidth=2)
    set_style(ax2, f'R² Score (best val = {best_val_r2:.4f})', 'R² Score', ylim=(-0.05, 1.05))
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df['epoch'], df['val_corr'], label='Val Correlation', color='tomato', linewidth=2)
    set_style(ax3, f'Correlation (best val = {best_val_corr:.4f})', 'Correlation', ylim=(-0.05, 1.05))
    
    # Bottom row (RMSE, MAE, Acc-like 20%)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(df['epoch'], df['val_rmse'], label='Val RMSE', color='tomato', linewidth=2)
    set_style(ax4, f'RMSE (best val = {best_val_rmse:.2f})', 'RMSE', ylim=(0, 2600))
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(df['epoch'], df['val_mae'], label='Val MAE', color='tomato', linewidth=2)
    set_style(ax5, f'MAE (best val = {best_val_mae:.2f})', 'MAE', ylim=(0, 2600))
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(df['epoch'], df['val_acc_like_0.2'], label='Val Acc-like (20%)', color='tomato', linewidth=2)
    set_style(ax6, f'Accuracy-like (20% tolerance) (best val = {best_val_acc_like:.4f})', 'Accuracy-like', ylim=(-0.05, 1.05))
    
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
    
    def plot_val_only_with_std(ax, epochs, val_mean, val_std, label='Val'):
        # Plot mean line
        ax.plot(epochs, val_mean, label=label, color='tomato', linewidth=2)
        
        # Add shaded area for std
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, 
                        color='tomato', alpha=0.2)
    
    # Calculate best validation values
    best_val_loss = df['val_loss'].min()
    best_val_r2 = df['val_r2'].max()
    best_val_corr = df['val_corr'].max()
    best_val_rmse = df['val_rmse'].min()
    best_val_mae = df['val_mae'].min()
    best_val_acc_like = df['val_acc_like_0.2'].max()
    
    # Top row (Loss, R², Correlation)
    ax1 = plt.subplot(2, 3, 1)
    plot_with_std(ax1, df['epoch'], df['train_loss'], df['val_loss'],
                  df_std['train_loss'], df_std['val_loss'], 'Loss')
    set_style(ax1, f'Loss (MSE) (best val = {best_val_loss:.4f})', 'Loss', ylim=(0, 1.1))
    
    ax2 = plt.subplot(2, 3, 2)
    plot_val_only_with_std(ax2, df['epoch'], df['val_r2'], df_std['val_r2'], 'Val R²')
    set_style(ax2, f'R² Score (best val = {best_val_r2:.4f})', 'R² Score', ylim=(-0.05, 1.05))
    
    ax3 = plt.subplot(2, 3, 3)
    plot_val_only_with_std(ax3, df['epoch'], df['val_corr'], df_std['val_corr'], 'Val Correlation')
    set_style(ax3, f'Correlation (best val = {best_val_corr:.4f})', 'Correlation', ylim=(-0.05, 1.05))
    
    # Bottom row (RMSE, MAE, Acc-like 20%)
    ax4 = plt.subplot(2, 3, 4)
    plot_val_only_with_std(ax4, df['epoch'], df['val_rmse'], df_std['val_rmse'], 'Val RMSE')
    set_style(ax4, f'RMSE (best val = {best_val_rmse:.2f})', 'RMSE', ylim=(0, 2600))
    
    ax5 = plt.subplot(2, 3, 5)
    plot_val_only_with_std(ax5, df['epoch'], df['val_mae'], df_std['val_mae'], 'Val MAE')
    set_style(ax5, f'MAE (best val = {best_val_mae:.2f})', 'MAE', ylim=(0, 2600))
    
    ax6 = plt.subplot(2, 3, 6)
    plot_val_only_with_std(ax6, df['epoch'], df['val_acc_like_0.2'], df_std['val_acc_like_0.2'], 'Val Acc-like (20%)')
    set_style(ax6, f'Accuracy-like (20% tolerance) (best val = {best_val_acc_like:.4f})', 'Accuracy-like', ylim=(-0.05, 1.05))
    
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
