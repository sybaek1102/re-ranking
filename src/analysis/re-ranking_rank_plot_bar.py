import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

# 3. 예측 label (10000, 2)의 마지막 열
oof_data = np.load(OOF_PATH)
pred_labels = oof_data['pred_label']  # (10000, 1)
print(f"  - Pred labels shape: {pred_labels.shape}")

# ======================== 정답 label=1인 인덱스 필터링 ========================
print("\n>>> 정답 label=1인 데이터 필터링...")
positive_mask = (true_labels == 1).flatten()
print(f"  - 정답 label=1 개수: {positive_mask.sum()}")

# 해당 인덱스로 데이터 추출
ranking_filtered = ranking[positive_mask]
pred_labels_filtered = pred_labels[positive_mask]

# ======================== 랭킹 검증 (랭킹=1 제외) ========================
print("\n>>> 랭킹 검증 (랭킹=1 제외)...")
valid_mask = (ranking_filtered != 1).flatten()
print(f"  - 랭킹=1인 데이터 개수: {(~valid_mask).sum()}")
print(f"  - 랭킹≠1인 데이터 개수: {valid_mask.sum()}")

# 랭킹≠1인 데이터만 사용
ranking_valid = ranking_filtered[valid_mask]
pred_labels_valid = pred_labels_filtered[valid_mask]

# ======================== TP / FN 분류 ========================
print("\n>>> TP / FN 분류...")
tp_mask = (pred_labels_valid == 1).flatten()
fn_mask = (pred_labels_valid == 0).flatten()

tp_ranking = ranking_valid[tp_mask]
fn_ranking = ranking_valid[fn_mask]

tp_count = tp_mask.sum()
fn_count = fn_mask.sum()

print(f"  - TP 개수: {tp_count}")
print(f"  - FN 개수: {fn_count}")

# ======================== 랭킹 분포 분석 ========================
print("\n>>> 랭킹 분포 분석...")

# 로그 스케일 범위 정의: 10^1 이하(2-10), 10^2 이하(11-100), 10^3 이하(101-1000)
bins = [2, 11, 101, 1001]
bin_labels = ['10^1 (2-10)', '10^2 (11-100)', '10^3 (101-1000)']

def compute_distribution(ranking_data, total_count, label="TP"):
    """랭킹 분포 계산 및 출력"""
    print(f"\n[{label} 분포 - 로그 스케일]")
    counts = []
    ratios = []
    
    for i in range(len(bins) - 1):
        mask = (ranking_data >= bins[i]) & (ranking_data < bins[i+1])
        count = mask.sum()
        ratio = (count / total_count * 100) if total_count > 0 else 0
        counts.append(count)
        ratios.append(ratio)
        
        # 기본 출력
        print(f"  {bin_labels[i]}: {count} ({ratio:.2f}%)", end='')
        
        # 10^3 구간일 경우 개별 rank 값들 출력
        if i == 2:  # 10^3 (101-1000) 구간
            if count > 0:
                # 해당 구간의 데이터만 추출
                range_data = ranking_data[mask]
                unique_ranks, unique_counts = np.unique(range_data, return_counts=True)
                
                # rank 값들을 문자열로 변환
                rank_details = ', '.join([f"{int(rank)}({cnt}개)" for rank, cnt in zip(unique_ranks, unique_counts)])
                print(f" (rank= {rank_details})")
            else:
                print()
        else:
            print()
    
    # 1000 이상
    above_1000 = (ranking_data >= 1001).sum()
    above_1000_ratio = (above_1000 / total_count * 100) if total_count > 0 else 0
    if above_1000 > 0:
        print(f"  1000+: {above_1000} ({above_1000_ratio:.2f}%)")
    
    # NaN 처리
    nan_count = np.isnan(ranking_data).sum()
    nan_ratio = (nan_count / total_count * 100) if total_count > 0 else 0
    if nan_count > 0:
        print(f"  NaN: {nan_count} ({nan_ratio:.2f}%)")
    
    return counts, ratios

# TP 분포
tp_counts, tp_ratios = compute_distribution(tp_ranking.flatten(), tp_count, "TP")

# FN 분포
fn_counts, fn_ratios = compute_distribution(fn_ranking.flatten(), fn_count, "FN")

# ======================== Linear scale 데이터 준비 ========================
linear_bins = [2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
linear_labels = ['2-10', '11-20', '21-30', '31-40', '41-50', 
                 '51-60', '61-70', '71-80', '81-90', '91-100']

def compute_linear_distribution(ranking_data, total_count):
    """10 단위 구간별 분포 계산"""
    counts = []
    ratios = []
    
    for i in range(len(linear_bins) - 1):
        mask = (ranking_data >= linear_bins[i]) & (ranking_data < linear_bins[i+1])
        count = mask.sum()
        ratio = (count / total_count * 100) if total_count > 0 else 0
        counts.append(count)
        ratios.append(ratio)
    
    return counts, ratios

tp_linear_counts, tp_linear_ratios = compute_linear_distribution(tp_ranking.flatten(), tp_count)
fn_linear_counts, fn_linear_ratios = compute_linear_distribution(fn_ranking.flatten(), fn_count)

# ======================== Rank 2-10 데이터 준비 ========================
def compute_top10_distribution(ranking_data, total_count):
    """Rank 2-10 개별 분포 계산"""
    counts = []
    ratios = []
    
    for rank in range(2, 11):
        count = (ranking_data == rank).sum()
        ratio = (count / total_count * 100) if total_count > 0 else 0
        counts.append(count)
        ratios.append(ratio)
    
    return counts, ratios

tp_top10_counts, tp_top10_ratios = compute_top10_distribution(tp_ranking.flatten(), tp_count)
fn_top10_counts, fn_top10_ratios = compute_top10_distribution(fn_ranking.flatten(), fn_count)
top10_labels = [str(i) for i in range(2, 11)]

# ======================== 막대 그래프 생성 ========================
print("\n>>> 막대 그래프 생성 중...")

# GridSpec을 사용하여 너비 비율 조정
# 세 그래프의 너비를 동일하게 설정
fig_bar = plt.figure(figsize=(30, 8))
gs = GridSpec(1, 3, figure=fig_bar, width_ratios=[1, 1, 1], wspace=0.15)
plt.rcParams.update({"font.size": 12})

# 모든 막대 너비를 동일하게 설정
bar_width = 0.4

# x축 데이터 준비
x_labels_log = bin_labels
x_pos_log = np.arange(len(x_labels_log))
x_pos_linear = np.arange(len(linear_labels))
x_pos_top10 = np.arange(len(top10_labels))

# --- 막대 그래프 1: 로그 스케일 ---
ax_bar1 = fig_bar.add_subplot(gs[0, 0])

bars_tp_log = ax_bar1.bar(x_pos_log - bar_width/2, tp_ratios, bar_width, 
                           color='royalblue', label=f'TP ({tp_count})')
bars_fn_log = ax_bar1.bar(x_pos_log + bar_width/2, fn_ratios, bar_width, 
                           color='tomato', label=f'FN ({fn_count})')

for i, (bar_tp, bar_fn) in enumerate(zip(bars_tp_log, bars_fn_log)):
    height_tp = bar_tp.get_height()
    count_tp = int(tp_ratios[i] * tp_count / 100)
    ax_bar1.text(bar_tp.get_x() + bar_tp.get_width()/2., height_tp,
                 f'{tp_ratios[i]:.1f}%\n({count_tp})',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    height_fn = bar_fn.get_height()
    count_fn = int(fn_ratios[i] * fn_count / 100)
    ax_bar1.text(bar_fn.get_x() + bar_fn.get_width()/2., height_fn,
                 f'{fn_ratios[i]:.1f}%\n({count_fn})',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

ax_bar1.set_xlabel('Ranking Range (Log Scale)', fontsize=14, fontweight='bold')
ax_bar1.set_ylabel('Ratio (%)', fontsize=14, fontweight='bold')
ax_bar1.set_title('Ranking Distribution: TP vs FN (Log Scale)', fontsize=16, fontweight='bold')
ax_bar1.set_xticks(x_pos_log)
ax_bar1.set_xticklabels(x_labels_log, rotation=15, ha='right')
ax_bar1.set_ylim(0, 100)
ax_bar1.grid(True, linestyle=':', alpha=0.6, axis='y')
ax_bar1.legend(fontsize=12, loc='best')

# --- 막대 그래프 2: Linear scale ---
ax_bar2 = fig_bar.add_subplot(gs[0, 1])

bars_tp_lin = ax_bar2.bar(x_pos_linear - bar_width/2, tp_linear_ratios, bar_width, 
                           color='royalblue', label=f'TP ({tp_count})')
bars_fn_lin = ax_bar2.bar(x_pos_linear + bar_width/2, fn_linear_ratios, bar_width, 
                           color='tomato', label=f'FN ({fn_count})')

# 모든 막대에 레이블 표시 (0 포함)
for i in range(len(bars_tp_lin)):
    height_tp = bars_tp_lin[i].get_height()
    ax_bar2.text(bars_tp_lin[i].get_x() + bars_tp_lin[i].get_width()/2., height_tp,
                 f'{tp_linear_ratios[i]:.1f}%\n({tp_linear_counts[i]})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    height_fn = bars_fn_lin[i].get_height()
    ax_bar2.text(bars_fn_lin[i].get_x() + bars_fn_lin[i].get_width()/2., height_fn,
                 f'{fn_linear_ratios[i]:.1f}%\n({fn_linear_counts[i]})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_bar2.set_xlabel('Ranking Range (10-unit intervals)', fontsize=14, fontweight='bold')
ax_bar2.set_ylabel('Ratio (%)', fontsize=14, fontweight='bold')
ax_bar2.set_title('Ranking Distribution: TP vs FN (Linear Scale)', fontsize=16, fontweight='bold')
ax_bar2.set_xticks(x_pos_linear)
ax_bar2.set_xticklabels(linear_labels, rotation=45, ha='right')
ax_bar2.set_ylim(0, 100)
ax_bar2.grid(True, linestyle=':', alpha=0.6, axis='y')
ax_bar2.legend(fontsize=12, loc='best')

# --- 막대 그래프 3: Rank 2-10 ---
ax_bar3 = fig_bar.add_subplot(gs[0, 2])

bars_tp_top = ax_bar3.bar(x_pos_top10 - bar_width/2, tp_top10_ratios, bar_width, 
                           color='royalblue', label=f'TP ({tp_count})')
bars_fn_top = ax_bar3.bar(x_pos_top10 + bar_width/2, fn_top10_ratios, bar_width, 
                           color='tomato', label=f'FN ({fn_count})')

# 모든 막대에 레이블 표시 (0 포함)
for i in range(len(bars_tp_top)):
    height_tp = bars_tp_top[i].get_height()
    ax_bar3.text(bars_tp_top[i].get_x() + bars_tp_top[i].get_width()/2., height_tp,
                 f'{tp_top10_ratios[i]:.1f}%\n({tp_top10_counts[i]})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    height_fn = bars_fn_top[i].get_height()
    ax_bar3.text(bars_fn_top[i].get_x() + bars_fn_top[i].get_width()/2., height_fn,
                 f'{fn_top10_ratios[i]:.1f}%\n({fn_top10_counts[i]})',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_bar3.set_xlabel('Ranking Position', fontsize=14, fontweight='bold')
ax_bar3.set_ylabel('Ratio (%)', fontsize=14, fontweight='bold')
ax_bar3.set_title('Ranking Distribution: TP vs FN (Rank 2-10)', fontsize=16, fontweight='bold')
ax_bar3.set_xticks(x_pos_top10)
ax_bar3.set_xticklabels(top10_labels)
ax_bar3.set_ylim(0, 100)
ax_bar3.grid(True, linestyle=':', alpha=0.6, axis='y')
ax_bar3.legend(fontsize=12, loc='best')

# 막대 그래프 저장
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12)
bar_png_path = os.path.join(RESULTS_DIR, "re-ranking_rank_plot_bar.png")
plt.savefig(bar_png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close(fig_bar)
print(f">>> 막대 그래프 저장 완료: {bar_png_path}")

# 10단위 분포 통계 출력
print("\n[TP 분포 - 10단위 구간]")
for i, label in enumerate(linear_labels):
    print(f"  {label}: {tp_linear_counts[i]} ({tp_linear_ratios[i]:.2f}%)")

print("\n[FN 분포 - 10단위 구간]")
for i, label in enumerate(linear_labels):
    print(f"  {label}: {fn_linear_counts[i]} ({fn_linear_ratios[i]:.2f}%)")

# Rank 2-10 개별 분포 통계 출력
print("\n[TP 분포 - Rank 2-10]")
for i, rank in enumerate(range(2, 11)):
    print(f"  Rank {rank}: {tp_top10_counts[i]} ({tp_top10_ratios[i]:.2f}%)")

print("\n[FN 분포 - Rank 2-10]")
for i, rank in enumerate(range(2, 11)):
    print(f"  Rank {rank}: {fn_top10_counts[i]} ({fn_top10_ratios[i]:.2f}%)")

print("\n" + "="*50)
print(">>> 분석 완료!")
print("="*50)
