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
df = pd.read_csv(csv_path)

# Derive output PNG path (same base name as CSV)
base_name = os.path.splitext(args.csv)[0]
png_path = os.path.join(RESULTS_DIR, f"{base_name}.png")

# -----------------------------------------------------------------------------
# [Step 1] Plotting (top 3 metrics, bottom 4 metrics)
# -----------------------------------------------------------------------------
plt.figure(figsize=(24, 12))
plt.rcParams.update({"font.size": 12})

# Helper for styling
def set_style(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

# Top row (Loss, Accuracy, AUC) – 2 rows, 3 columns layout
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

# Bottom row (Precision & Recall for both classes) – 2 rows, 4 columns layout
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

# Layout and save
plt.tight_layout()
plt.savefig(png_path, dpi=300)

print("=" * 50)
print(f">>> Plot saved to: {png_path}")
print("=" * 50)
