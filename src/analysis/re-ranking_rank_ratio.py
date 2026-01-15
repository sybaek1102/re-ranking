import os
import numpy as np
import matplotlib.pyplot as plt

# ======================== 경로 설정 ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../../results/analysis")

# 입력 파일 경로
RANK_PATH = os.path.join(INPUT_DIR, "re-ranking_analysis_rank.npz")
FEATURES_PATH = os.path.join(INPUT_DIR, "re-ranking_features.npz")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof", "re-ranking_mlp_oof.npz")

# 출력 파일 경로
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_GRAPH_PATH = os.path.join(RESULTS_DIR, "re-ranking_rank_ratio.png")

# ======================== 데이터 로드 ========================
print(">>> 데이터 로딩 중...")

# 1. 랭킹 데이터 (10000, 1)
rank_data = np.load(RANK_PATH)
ranking = rank_data['positions']  # (10000, 1)
print(f"  - Ranking shape: {ranking.shape}")

# 2. 정답 label (10000, 17)의 마지막 열
features_data = np.load(FEATURES_PATH)
dataset = features_data['data']
true_labels = dataset[:, -1].reshape(-1, 1)  # (10000, 1)
print(f"  - True labels shape: {true_labels.shape}")

# ======================== 정답 label=1인 인덱스 필터링 ========================
print("\n>>> 정답 label=1인 데이터 필터링...")
positive_mask = (true_labels == 1).flatten()
print(f"  - 정답 label=1 개수: {positive_mask.sum()}")

# 해당 인덱스로 데이터 추출
ranking_filtered = ranking[positive_mask]

# ======================== 랭킹 검증 (랭킹=1 제외) ========================
print("\n>>> 랭킹 검증 (랭킹=1 제외)...")
valid_mask = (ranking_filtered != 1).flatten()
print(f"  - 랭킹=1인 데이터 개수: {(~valid_mask).sum()}")
print(f"  - 랭킹≠1인 데이터 개수: {valid_mask.sum()}")

# 랭킹≠1인 데이터만 사용 (전체 TP+FN)
ranking_valid = ranking_filtered[valid_mask]
total_count = len(ranking_valid)

print(f"\n>>> total data (TP+FN): {total_count}")

# ======================== 랭킹 분포 분석 ========================
print("\n>>> 랭킹 분포 분석...")

# 로그 스케일 범위 정의: 10^1 이하(2-10), 10^2 이하(11-100), 10^3 이하(101-1000)
bins = [2, 11, 101, 1001]
bin_labels = ['10^1 (2-10)', '10^2 (11-100)', '10^3 (101-1000)']

counts = []
ratios = []

print("\n[전체 분포 - 로그 스케일]")
for i in range(len(bins) - 1):
    mask = (ranking_valid >= bins[i]) & (ranking_valid < bins[i+1])
    count = mask.sum()
    ratio = (count / total_count * 100) if total_count > 0 else 0
    counts.append(count)
    ratios.append(ratio)
    
    print(f"  {bin_labels[i]}: {count} ({ratio:.2f}%)")

# 1000 이상
above_1000 = (ranking_valid >= 1001).sum()
above_1000_ratio = (above_1000 / total_count * 100) if total_count > 0 else 0
if above_1000 > 0:
    print(f"  1000+: {above_1000} ({above_1000_ratio:.2f}%)")

# ======================== 그래프 생성 ========================
print("\n>>> 그래프 생성 중...")

fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({"font.size": 12})

x_pos = np.arange(len(bin_labels))
bar_width = 0.5

# 막대 그래프
bars = ax.bar(x_pos, ratios, bar_width, color='steelblue', label=f'total (TP+FN) ({total_count})')

# 막대 위에 레이블 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratios[i]:.2f}%\n({counts[i]})',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Ranking Range (Log Scale)', fontsize=14, fontweight='bold')
ax.set_ylabel('Ratio (%)', fontsize=14, fontweight='bold')
ax.set_title('Overall Ranking Distribution (TP+FN Combined)', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, rotation=15, ha='right')
ax.set_ylim(0, 100)
ax.grid(True, linestyle=':', alpha=0.6, axis='y')
ax.legend(fontsize=12, loc='best')

# 그래프 저장
plt.tight_layout()
plt.savefig(OUTPUT_GRAPH_PATH, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f">>> 그래프 저장 완료: {OUTPUT_GRAPH_PATH}")

print("\n" + "="*50)
print(">>> 분석 완료!")
print("="*50)