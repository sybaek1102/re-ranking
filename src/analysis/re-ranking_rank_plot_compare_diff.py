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
OUTPUT_GRAPH_PATH = os.path.join(RESULTS_DIR, "re-ranking_rank_plot_compare_diff.png")

# ======================== 데이터 로드 & 전처리 (Silent) ========================
# 1. 데이터 로드
rank_data = np.load(RANK_PATH)
ranking = rank_data['positions']

features_data = np.load(FEATURES_PATH)
dataset = features_data['data']
true_labels = dataset[:, -1].reshape(-1, 1)

oof_data = np.load(OOF_PATH)
pred_labels = oof_data['pred_label']

# 2. 정답 label=1인 인덱스 필터링
positive_mask = (true_labels == 1).flatten()
ranking_filtered = ranking[positive_mask]
pred_labels_filtered = pred_labels[positive_mask]

# 3. 랭킹 검증 (랭킹=1 제외)
valid_mask = (ranking_filtered != 1).flatten()

# Total 데이터 (TP + FN)
ranking_valid = ranking_filtered[valid_mask]
pred_labels_valid = pred_labels_filtered[valid_mask]
total_count = len(ranking_valid)

# 4. TP / FN 분류
tp_mask = (pred_labels_valid == 1).flatten()
fn_mask = (pred_labels_valid == 0).flatten()

tp_ranking = ranking_valid[tp_mask]
fn_ranking = ranking_valid[fn_mask]

tp_count = tp_mask.sum()
fn_count = fn_mask.sum()

# ======================== 랭킹 분포 계산 (Exact Rank 2~1000) ========================
# X축: 2부터 1000까지의 모든 Rank
x_ranks = np.arange(2, 1001)

def get_exact_ratios(ranking_data, total_count, target_ranks):
    """각 Rank 별 정확한 비율 계산"""
    if total_count == 0:
        return np.zeros(len(target_ranks))
    
    # 데이터에 존재하는 Rank와 개수를 센다
    unique_ranks, counts = np.unique(ranking_data, return_counts=True)
    rank_count_map = dict(zip(unique_ranks, counts))
    
    # target_ranks(2~1000)에 해당하는 값만 추출하여 비율 계산
    ratios = []
    for rank in target_ranks:
        count = rank_count_map.get(rank, 0)
        ratios.append(count / total_count * 100)
        
    return np.array(ratios)

# 각 그룹별 비율 계산
tp_ratios = get_exact_ratios(tp_ranking, tp_count, x_ranks)
fn_ratios = get_exact_ratios(fn_ranking, fn_count, x_ranks)
total_ratios = get_exact_ratios(ranking_valid, total_count, x_ranks)

# ======================== 상대적 차이 계산 (핵심 변경 사항) ========================
# Total 분포를 0으로 기준 잡기 위해 뺄셈 연산 수행
diff_tp = tp_ratios - total_ratios
diff_fn = fn_ratios - total_ratios

# ======================== 그래프 생성 (Log X Scale) ========================
plt.figure(figsize=(10, 7))
plt.rcParams.update({"font.size": 12})

# 1. Baseline (Total) - 기준선 0
plt.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.8, label='Total (Baseline)')

# 2. TP Line (Blue) - Relative to Total
plt.plot(x_ranks, diff_tp, color='blue', linewidth=1.5, alpha=0.8,
         label=f'TP - Total')

# 3. FN Line (Red) - Relative to Total
plt.plot(x_ranks, diff_fn, color='red', linewidth=1.5, alpha=0.8,
         label=f'FN - Total')

# X축 로그 스케일 설정 (Base 2)
plt.xscale('log', base=2)

# X축 틱 설정
ticks = [2**i for i in range(0, 11)] # 1, 2, 4 ... 1024
plt.xticks(ticks, ticks)
plt.xlim(1.8, 1000)

# Y축 범위 설정 (데이터에 따라 자동 조정되지만, 0이 중앙에 오도록 약간의 여유를 둠)
# 필요시 ylim을 직접 지정하세요 (예: plt.ylim(-5, 5))

# 그래프 스타일 설정
plt.xlabel('gt ranking position (Log Scale)', fontsize=14, fontweight='bold')
plt.ylabel('Difference from Total (%)', fontsize=14, fontweight='bold') # Y축 라벨 변경
plt.title('Relative Ranking Distribution (TP/FN vs Total)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.4)
plt.legend(fontsize=12, loc='upper right')

# 저장
plt.savefig(OUTPUT_GRAPH_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"Graph saved to {OUTPUT_GRAPH_PATH}")