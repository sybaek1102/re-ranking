import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# [Step 0] 경로 설정 및 데이터 로드
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

STATUS_PATH = os.path.join(INPUT_DIR, "re-ranking_status_label.npz")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof", "21_re-ranking_pqD_pred_resD_oof.npz")
SAVE_FIG_PATH = "/home/syback/vectorDB/re-ranking/results/analysis/re-ranking_mlp_pqD_pred_resD.png"

# 데이터 로드
status_data = np.load(STATUS_PATH)['status_label'].flatten()
oof_data = np.load(OOF_PATH)['pred_label'].flatten().astype(int)
total_count = len(status_data)

# -----------------------------------------------------------------------------
# [Step 1] 색상 정의 (이미지 기준)
# -----------------------------------------------------------------------------
COLOR_RED = '#d62728'       # 빨간색 (Positive / State 1)
COLOR_BLUE = '#377eb8'      # 파란색 (Negative / State 0)
COLOR_L_BLUE = '#aec7e8'    # 하늘색 (State -1)

# -----------------------------------------------------------------------------
# [Step 2] 통계 계산
# -----------------------------------------------------------------------------
# 1. Binary
binary_labels = np.where(status_data == 1, 1, 0)
b_counts = np.bincount(binary_labels, minlength=2)
b_ratios = b_counts / total_count

# 2. Status (-1, 0, 1)
s_counts = np.bincount(status_data + 1, minlength=3)
s_ratios = s_counts / total_count

# 3. Correct Cases
c_a = np.sum((oof_data == 0) & (status_data == -1)) / total_count
c_b = np.sum((oof_data == 0) & (status_data == 0)) / total_count
c_c = np.sum((oof_data == 1) & (status_data == 1)) / total_count

# 4. Wrong Cases
w_d = np.sum((oof_data == 1) & (status_data == -1)) / total_count
w_e = np.sum((oof_data == 1) & (status_data == 0)) / total_count
w_f = np.sum((oof_data == 0) & (status_data == 1)) / total_count

# -----------------------------------------------------------------------------
# [Step 3] 그래프 그리기
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- 1. Binary Distribution (2 bars: Blue, Red) ---
axes[0, 0].bar(['Label 0\n(Neg)', 'Label 1\n(Pos)'], b_ratios, color=[COLOR_BLUE, COLOR_RED])
axes[0, 0].set_title("1. Total Label Dist (N=10000)", fontsize=13, fontweight='bold')

# --- 2. Total State Distribution (3 bars: L_Blue, Blue, Red) ---
axes[0, 1].bar(['State -1', 'State 0', 'State 1'], s_ratios, color=[COLOR_L_BLUE, COLOR_BLUE, COLOR_RED])
axes[0, 1].set_title("2. Total State Dist (N=10000)", fontsize=13, fontweight='bold')

# --- 3. Correct Predictions (3 bars: L_Blue, Blue, Red) ---
labels_3 = ['P:0 & S:-1', 'P:0 & S:0', 'P:1 & S:1']
axes[1, 0].bar(labels_3, [c_a, c_b, c_c], color=[COLOR_L_BLUE, COLOR_BLUE, COLOR_RED])
axes[1, 0].set_title("3. Correct Prediction Ratio (Total Base)", fontsize=13, fontweight='bold')

# --- 4. Wrong Predictions (3 bars: L_Blue, Blue, Red) ---
labels_4 = ['P:1 & S:-1', 'P:1 & S:0', 'P:0 & S:1']
axes[1, 1].bar(labels_4, [w_d, w_e, w_f], color=[COLOR_L_BLUE, COLOR_BLUE, COLOR_RED])
axes[1, 1].set_title("4. Wrong Prediction Ratio (Total Base)", fontsize=13, fontweight='bold')

# 공통 설정 (Y축 범위 및 수치 표시)
for ax in axes.flat:
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Ratio (%)", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold', fontsize=11)

# 저장 및 출력
save_dir = os.path.dirname(SAVE_FIG_PATH)
if not os.path.exists(save_dir): os.makedirs(save_dir)

plt.savefig(SAVE_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"\n>>> 시각화 완료: {SAVE_FIG_PATH}")
plt.show()