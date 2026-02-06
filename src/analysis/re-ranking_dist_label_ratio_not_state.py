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

# Feature 파일에서 label 추출 (마지막 열)
FEATURE_PATH = os.path.join(INPUT_DIR, "21_re-ranking_pqD_pred_resD_label.npz")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof", "21_re-ranking_mlp_pqD_pred_resD_label_oof.npz")
SAVE_FIG_PATH = "/home/syback/vectorDB/re-ranking/results/analysis/re-ranking_mlp_pqD_pred_resD_label_not_state.png"

print("="*70)
print(">>> Label-based Prediction Analysis (Without Status File)")
print("="*70)

# Feature 파일 로드 및 label 추출
print(f"\n[1/3] Feature 파일 로드 중...")
print(f"    파일: {FEATURE_PATH}")
feature_data = np.load(FEATURE_PATH)['data']  # (10000, 33)
print(f"    Shape: {feature_data.shape}")

# 마지막 열이 label (0 또는 1)
true_labels = feature_data[:, -1].flatten().astype(int)  # (10000,)
print(f"    Label Shape: {true_labels.shape}")
print(f"    Label 값 범위: {np.min(true_labels)} ~ {np.max(true_labels)}")
print(f"    Label 0 개수: {np.sum(true_labels == 0)}")
print(f"    Label 1 개수: {np.sum(true_labels == 1)}")

# OOF 예측값 로드
print(f"\n[2/3] OOF 예측값 로드 중...")
print(f"    파일: {OOF_PATH}")
oof_data = np.load(OOF_PATH)['pred_label'].flatten().astype(int)  # (10000,)
print(f"    Prediction Shape: {oof_data.shape}")
print(f"    Prediction 값 범위: {np.min(oof_data)} ~ {np.max(oof_data)}")
print(f"    Prediction 0 개수: {np.sum(oof_data == 0)}")
print(f"    Prediction 1 개수: {np.sum(oof_data == 1)}")

total_count = len(true_labels)

# -----------------------------------------------------------------------------
# [Step 1] 색상 정의
# -----------------------------------------------------------------------------
COLOR_RED = '#d62728'       # 빨간색 (Positive / Label 1)
COLOR_BLUE = '#377eb8'      # 파란색 (Negative / Label 0)
COLOR_GREEN = '#2ca02c'     # 초록색 (Correct)
COLOR_ORANGE = '#ff7f0e'    # 주황색 (Wrong)

# -----------------------------------------------------------------------------
# [Step 2] 통계 계산
# -----------------------------------------------------------------------------
print(f"\n[3/3] 통계 계산 중...")

# 1. True Label Distribution
true_label_counts = np.bincount(true_labels, minlength=2)
true_label_ratios = true_label_counts / total_count

# 2. Predicted Label Distribution
pred_label_counts = np.bincount(oof_data, minlength=2)
pred_label_ratios = pred_label_counts / total_count

# 3. Correct vs Wrong
correct_mask = (oof_data == true_labels)
wrong_mask = ~correct_mask

correct_ratio = np.sum(correct_mask) / total_count
wrong_ratio = np.sum(wrong_mask) / total_count

# 4. Detailed Analysis
# Correct Cases
correct_0 = np.sum((oof_data == 0) & (true_labels == 0)) / total_count  # True Negative
correct_1 = np.sum((oof_data == 1) & (true_labels == 1)) / total_count  # True Positive

# Wrong Cases
wrong_0_to_1 = np.sum((oof_data == 1) & (true_labels == 0)) / total_count  # False Positive
wrong_1_to_0 = np.sum((oof_data == 0) & (true_labels == 1)) / total_count  # False Negative

print(f"    - 전체 정확도: {correct_ratio*100:.2f}%")
print(f"    - True Negative (P:0 & L:0): {correct_0*100:.2f}%")
print(f"    - True Positive (P:1 & L:1): {correct_1*100:.2f}%")
print(f"    - False Positive (P:1 & L:0): {wrong_0_to_1*100:.2f}%")
print(f"    - False Negative (P:0 & L:1): {wrong_1_to_0*100:.2f}%")

# -----------------------------------------------------------------------------
# [Step 3] 그래프 그리기
# -----------------------------------------------------------------------------
print(f"\n>>> 그래프 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# --- 1. True Label Distribution ---
axes[0, 0].bar(['Label 0\n(Negative)', 'Label 1\n(Positive)'], 
               true_label_ratios, color=[COLOR_BLUE, COLOR_RED])
axes[0, 0].set_title("1. True Label Distribution (N=10000)", fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel("Ratio", fontsize=11)

# --- 2. Predicted Label Distribution ---
axes[0, 1].bar(['Pred 0\n(Negative)', 'Pred 1\n(Positive)'], 
               pred_label_ratios, color=[COLOR_BLUE, COLOR_RED])
axes[0, 1].set_title("2. Predicted Label Distribution (N=10000)", fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel("Ratio", fontsize=11)

# --- 3. Overall Accuracy ---
axes[0, 2].bar(['Correct', 'Wrong'], 
               [correct_ratio, wrong_ratio], color=[COLOR_GREEN, COLOR_ORANGE])
axes[0, 2].set_title("3. Overall Prediction Accuracy", fontsize=13, fontweight='bold')
axes[0, 2].set_ylabel("Ratio", fontsize=11)

# --- 4. Correct Predictions Detail ---
labels_correct = ['True Negative\n(P:0 & L:0)', 'True Positive\n(P:1 & L:1)']
axes[1, 0].bar(labels_correct, [correct_0, correct_1], color=[COLOR_BLUE, COLOR_RED])
axes[1, 0].set_title("4. Correct Predictions (Total Base)", fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel("Ratio", fontsize=11)

# --- 5. Wrong Predictions Detail ---
labels_wrong = ['False Positive\n(P:1 & L:0)', 'False Negative\n(P:0 & L:1)']
axes[1, 1].bar(labels_wrong, [wrong_0_to_1, wrong_1_to_0], color=[COLOR_ORANGE, COLOR_ORANGE])
axes[1, 1].set_title("5. Wrong Predictions (Total Base)", fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel("Ratio", fontsize=11)

# --- 6. Confusion Matrix Style ---
confusion_data = [
    [correct_0, wrong_0_to_1],  # True Label 0
    [wrong_1_to_0, correct_1]   # True Label 1
]

im = axes[1, 2].imshow(confusion_data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[1, 2].set_xticks([0, 1])
axes[1, 2].set_yticks([0, 1])
axes[1, 2].set_xticklabels(['Pred 0', 'Pred 1'], fontsize=11)
axes[1, 2].set_yticklabels(['True 0', 'True 1'], fontsize=11)
axes[1, 2].set_title("6. Confusion Matrix (Ratio)", fontsize=13, fontweight='bold')

# Confusion matrix 값 표시
for i in range(2):
    for j in range(2):
        text = axes[1, 2].text(j, i, f'{confusion_data[i][j]*100:.2f}%',
                               ha="center", va="center", color="black", 
                               fontsize=14, fontweight='bold')

# Colorbar 추가
cbar = plt.colorbar(im, ax=axes[1, 2])
cbar.set_label('Ratio', fontsize=11)

# 공통 설정 (Y축 범위 및 수치 표시) - Confusion Matrix 제외
for i in range(2):
    for j in range(3):
        if i == 1 and j == 2:  # Confusion Matrix 제외
            continue
        ax = axes[i, j]
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height*100:.2f}%', 
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='center', xytext=(0, 10), 
                       textcoords='offset points', fontweight='bold', fontsize=11)

# 저장 및 출력
save_dir = os.path.dirname(SAVE_FIG_PATH)
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)

plt.savefig(SAVE_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"\n>>> 시각화 완료: {SAVE_FIG_PATH}")
print("="*70)