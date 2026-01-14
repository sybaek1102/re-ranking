import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve
import os

# -----------------------------------------------------------------------------
# [Step 0] 설정: 경로 및 파라미터
# -----------------------------------------------------------------------------
# 파일 경로 설정 (상대 경로 사용)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

INPUT_PATH = os.path.join(INPUT_DIR, "re-ranking_features.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "re-ranking_mlp.csv")

# 하이퍼파라미터
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 50
VAL_RATIO = 0.1
THRESHOLD = 0.5  # 필요하다면 최적값으로 변경하세요

# -----------------------------------------------------------------------------
# [Step 1] 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
print(">>> 데이터 로딩 중...")
if not os.path.exists(INPUT_PATH):
    print(f"파일이 존재하지 않습니다: {INPUT_PATH}")
    exit()

data = np.load(INPUT_PATH)
dataset = data['data'] 

# Feature(X)와 Label(y) 분리
X_numpy = dataset[:, :-1].astype(np.float32)
y_numpy = dataset[:, -1].astype(np.float32).reshape(-1, 1)

# Train / Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_numpy, y_numpy, test_size=VAL_RATIO, random_state=42, stratify=y_numpy
)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

print(f"Train Set: {X_train.shape}, Val Set: {X_val.shape}")

# -----------------------------------------------------------------------------
# [Step 2] 모델 정의
# -----------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = SimpleMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------------------------------------------------------
# [Step 3] 학습 Loop 및 로그 기록
# -----------------------------------------------------------------------------
history = [] 

print(f">>> 학습 시작 (Total Epochs: {EPOCHS}, Threshold: {THRESHOLD})")

for epoch in range(1, EPOCHS + 1):
    # --- [Training Phase] ---
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0
    
    train_probs_list = []
    train_labels_list = []
    
    for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        train_probs_list.append(outputs.detach().cpu().numpy())
        train_labels_list.append(batch_y.detach().cpu().numpy())

    # Train Metrics
    train_probs_concat = np.concatenate(train_probs_list)
    train_labels_concat = np.concatenate(train_labels_list)
    avg_train_loss = epoch_loss / (len(X_train_tensor) // BATCH_SIZE)
    
    train_auc = roc_auc_score(train_labels_concat, train_probs_concat)
    train_preds = (train_probs_concat >= THRESHOLD).astype(int)
    train_acc = accuracy_score(train_labels_concat, train_preds)
    
    # Train Precision & Recall (Class 0, 1 각각)
    tr_prec, tr_rec, _, _ = precision_recall_fscore_support(
        train_labels_concat, train_preds, average=None, zero_division=0
    )

    # --- [Validation Phase] ---
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
        
        val_probs = val_outputs.cpu().numpy()
        val_labels = y_val_tensor.cpu().numpy()
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_preds = (val_probs >= THRESHOLD).astype(int)
        val_acc = accuracy_score(val_labels, val_preds)

        # Validation Precision & Recall (Class 0, 1 각각)
        val_prec, val_rec, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average=None, zero_division=0
        )

    # --- [Logging] ---
    log_entry = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_prec0": tr_prec[0],
        "val_prec0": val_prec[0],
        "train_prec1": tr_prec[1],
        "val_prec1": val_prec[1],
        "train_rec0": tr_rec[0],
        "val_rec0": val_rec[0],
        "train_rec1": tr_rec[1],
        "val_rec1": val_rec[1],
    }
    history.append(log_entry)

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{EPOCHS}] Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val P1: {val_prec[1]:.4f} R1: {val_rec[1]:.4f}")

# -----------------------------------------------------------------------------
# [Step 4] 로그 파일 저장
# -----------------------------------------------------------------------------
# 출력 디렉토리 생성
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

df_history = pd.DataFrame(history)

# 저장할 컬럼 순서 지정 (residual-sign_mlp.py와 동일한 형식)
cols = [
    "epoch", 
    "train_loss", "val_loss",
    "train_acc", "val_acc",
    "train_auc", "val_auc",
    "train_prec0", "val_prec0",
    "train_prec1", "val_prec1",
    "train_rec0", "val_rec0",
    "train_rec1", "val_rec1"
]
df_history = df_history[cols]

df_history.to_csv(LOG_PATH, index=False)

print("\n" + "="*50)
print(f">>> 학습 종료.")
print(f">>> 상세 로그 저장 완료: {LOG_PATH}")
print("="*50)

# 최종 확인
print("\n[최종 Validation 상세 성능]")
last_log = history[-1]
print(f"Accuracy : {last_log['val_acc']:.4f}")
print(f"AUC      : {last_log['val_auc']:.4f}")
print("-" * 30)
print(f"Class 0 (Negative) | Precision: {last_log['val_prec0']:.4f}, Recall: {last_log['val_rec0']:.4f}")
print(f"Class 1 (Positive) | Precision: {last_log['val_prec1']:.4f}, Recall: {last_log['val_rec1']:.4f}")
print("="*50)
