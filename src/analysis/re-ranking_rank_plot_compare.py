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
OUTPUT_GRAPH_PATH = os.path.join(RESULTS_DIR, "re-ranking_rank_plot_compare.png")

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

# ======================== 그래프 생성 (Log X Scale) ========================
plt.figure(figsize=(10, 7))  # X축이 넓으므로 가로 길이 증가
plt.rcParams.update({"font.size": 12})

# 1. TP Line (Blue)
plt.plot(x_ranks, tp_ratios, color='blue', linewidth=1.5, alpha=0.8,
         label=f'TP ({tp_count})')

# 2. FN Line (Red)
plt.plot(x_ranks, fn_ratios, color='red', linewidth=1.5, alpha=0.8,
         label=f'FN ({fn_count})')

# 3. Total Line (Green)
plt.plot(x_ranks, total_ratios, color='gray', linewidth=1.5, linestyle='--', alpha=0.9,
         label=f'Total ({total_count})')

# X축 로그 스케일 설정 (Base 2)
plt.xscale('log', base=2)

# X축 틱 설정 (2, 4, 8, ... 1024)
ticks = [2**i for i in range(0, 11)] # 2, 4, 8 ... 1024
plt.xticks(ticks, ticks) # 틱 위치와 레이블을 동일하게 설정
plt.xlim(1.8, 1000) # 2~1000 범위 고정
plt.ylim(-1.0, 50)    

# 그래프 스타일 설정
plt.xlabel('gt ranking position', fontsize=14, fontweight='bold')
plt.ylabel('Ratio (%)', fontsize=14, fontweight='bold')
plt.title('Ranking Distribution: Exact Rank (2-1000)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.4) # 격자 좀 더 촘촘하게
plt.legend(fontsize=12, loc='upper right')

# 저장
plt.savefig(OUTPUT_GRAPH_PATH, dpi=300, bbox_inches='tight')
plt.close()