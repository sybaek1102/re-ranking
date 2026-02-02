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

SAVE_FIG_PATH = "/home/syback/vectorDB/re-ranking/results/analysis/re-ranking_1xN_label_ratio.png"

# X축 레이블
x_labels = ['1x16', '1x32', '1x64', '1x128', '1x256', '1x512']

# -----------------------------------------------------------------------------
# [Step 1] 색상 정의
# -----------------------------------------------------------------------------
COLOR_RED = '#d62728'       # 빨간색 (State 1)
COLOR_BLUE = '#377eb8'      # 파란색 (State 0)
COLOR_L_BLUE = '#aec7e8'    # 하늘색 (State -1)

# -----------------------------------------------------------------------------
# [Step 2] 데이터 로드 및 통계 계산
# -----------------------------------------------------------------------------
state_minus1_ratios = []
state_0_ratios = []
state_1_ratios = []

for file_path in FILE_PATHS:
    # 데이터 로드
    status_data = np.load(file_path)['status_label'].flatten()
    total_count = len(status_data)
    
    # State별 개수 계산 (-1, 0, 1)
    s_counts = np.bincount(status_data + 1, minlength=3)
    s_ratios = (s_counts / total_count) * 100  # 퍼센트로 변환
    
    # 각 state별 비율 저장
    state_minus1_ratios.append(s_ratios[0])  # State -1
    state_0_ratios.append(s_ratios[1])        # State 0
    state_1_ratios.append(s_ratios[2])        # State 1

# -----------------------------------------------------------------------------
# [Step 3] 그래프 그리기
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

# 꺾은선 그래프 그리기
ax.plot(x_labels, state_minus1_ratios, marker='o', linewidth=2.5, markersize=10, 
        color=COLOR_L_BLUE, label='State -1', linestyle='-')
ax.plot(x_labels, state_0_ratios, marker='s', linewidth=2.5, markersize=10, 
        color=COLOR_BLUE, label='State 0', linestyle='-')
ax.plot(x_labels, state_1_ratios, marker='^', linewidth=2.5, markersize=10, 
        color=COLOR_RED, label='State 1', linestyle='-')

# 그래프 설정
ax.set_title("State Ratio Changes across Different Configurations", fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel("Configuration", fontsize=13, fontweight='bold')
ax.set_ylabel("Ratio (%)", fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)

# 각 데이터 포인트에 수치 표시
for i, x_label in enumerate(x_labels):
    ax.annotate(f'{state_minus1_ratios[i]:.2f}%', 
                (i, state_minus1_ratios[i]), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=9, fontweight='bold')
    ax.annotate(f'{state_0_ratios[i]:.2f}%', 
                (i, state_0_ratios[i]), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=9, fontweight='bold')
    ax.annotate(f'{state_1_ratios[i]:.2f}%', 
                (i, state_1_ratios[i]), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=9, fontweight='bold')

# 저장 및 출력
save_dir = os.path.dirname(SAVE_FIG_PATH)
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)

plt.tight_layout()
plt.savefig(SAVE_FIG_PATH, dpi=300, bbox_inches='tight')
print(f"\n>>> 시각화 완료: {SAVE_FIG_PATH}")
print(f"\n[State -1 Ratios]: {state_minus1_ratios}")
print(f"[State 0 Ratios]: {state_0_ratios}")
print(f"[State 1 Ratios]: {state_1_ratios}")
plt.show()