import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# [Step 0] 경로 설정 및 데이터 로드
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")

# 6개 파일 경로
FILE_PATHS = [
    os.path.join(INPUT_DIR, "re-ranking_status_label.npz"),
    os.path.join(INPUT_DIR, "re-ranking_status_label_1x32.npz"),
    os.path.join(INPUT_DIR, "re-ranking_status_label_1x64.npz"),
    os.path.join(INPUT_DIR, "re-ranking_status_label_1x128.npz"),
    os.path.join(INPUT_DIR, "re-ranking_status_label_1x256.npz"),
    os.path.join(INPUT_DIR, "re-ranking_status_label_1x512.npz")
]

SAVE_FIG_PATH = "/home/syback/vectorDB/re-ranking/results/analysis/re-ranking_1xN_label_ratio_stick.png"

# 레이블
config_labels = ['1x16', '1x32', '1x64', '1x128', '1x256', '1x512']
state_labels = ['State -1', 'State 0', 'State 1']

# -----------------------------------------------------------------------------
# [Step 1] 색상 정의 (각 configuration별로 다른 색상)
# -----------------------------------------------------------------------------
COLOR_1X16 = '#ff7f0e'    # 주황색
COLOR_1X32 = '#2ca02c'    # 초록색
COLOR_1X64 = '#1f77b4'    # 파란색
COLOR_1X128 = '#d62728'   # 빨간색
COLOR_1X256 = '#9467bd'   # 보라색
COLOR_1X512 = '#8c564b'   # 갈색

colors = [COLOR_1X16, COLOR_1X32, COLOR_1X64, COLOR_1X128, COLOR_1X256, COLOR_1X512]

# -----------------------------------------------------------------------------
# [Step 2] 데이터 로드 및 통계 계산
# -----------------------------------------------------------------------------
# 각 configuration별 state 비율 저장
data_matrix = []  # shape: (6 configs, 3 states)

for file_path in FILE_PATHS:
    # 데이터 로드
    status_data = np.load(file_path)['status_label'].flatten()
    total_count = len(status_data)
    
    # State별 개수 계산 (-1, 0, 1)
    s_counts = np.bincount(status_data + 1, minlength=3)
    s_ratios = (s_counts / total_count) * 100  # 퍼센트로 변환
    
    data_matrix.append(s_ratios)

data_matrix = np.array(data_matrix)  # shape: (6, 3)

# -----------------------------------------------------------------------------
# [Step 3] 막대 그래프 그리기
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(state_labels))  # State -1, 0, 1의 위치
width = 0.12  # 막대 너비 (6개를 배치하기 위해 줄임)

# 각 configuration별로 막대 그리기
for i, config in enumerate(config_labels):
    offset = width * (i - 2.5)  # 막대 위치 조정 (중앙 정렬)
    bars = ax.bar(x + offset, data_matrix[i], width, label=config, color=colors[i])
    
    # 각 막대 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

# 그래프 설정
ax.set_title("State Distribution Comparison across Configurations", fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel("State", fontsize=13, fontweight='bold')
ax.set_ylabel("Ratio (%)", fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(state_labels, fontsize=12)
ax.legend(fontsize=11, loc='best', frameon=True, shadow=True, ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 저장 및 출력
save_dir = os.path.dirname(SAVE_FIG_PATH)
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)

plt.tight_layout()
plt.savefig(SAVE_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"\n>>> 시각화 완료: {SAVE_FIG_PATH}")
print(f"\n[Data Matrix]")
print(f"{'Config':<10} {'State -1':<12} {'State 0':<12} {'State 1':<12}")
print("-" * 50)
for i, config in enumerate(config_labels):
    print(f"{config:<10} {data_matrix[i][0]:<12.2f} {data_matrix[i][1]:<12.2f} {data_matrix[i][2]:<12.2f}")
plt.show()