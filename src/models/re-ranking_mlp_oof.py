import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
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
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "re-ranking_mlp_oof.csv")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof", "re-ranking_mlp_oof.npz")

# 하이퍼파라미터
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 100        # 최대 Epoch
THRESHOLD = 0.5
NUM_FOLDS = 10          # 10-Fold

# -----------------------------------------------------------------------------
# [Step 1] 데이터 로드
# -----------------------------------------------------------------------------
print(">>> 데이터 로딩 중...")
if not os.path.exists(INPUT_PATH):
    print(f"파일이 존재하지 않습니다: {INPUT_PATH}")
    exit()

data = np.load(INPUT_PATH)
dataset = data['data'] 

X_numpy = dataset[:, :-1].astype(np.float32)
y_numpy = dataset[:, -1].astype(np.float32).reshape(-1, 1)
print(f"  - Feature Shape: {X_numpy.shape}")
print(f"  - Label Shape: {y_numpy.shape}")

# 전체 데이터 개수 및 인덱스 생성
num_samples = len(X_numpy)
all_indices = np.arange(num_samples)

# OOF 예측 결과를 담을 배열 초기화 (확률값 저장용)
oof_probs = np.zeros((num_samples, 1), dtype=np.float32)

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

# -----------------------------------------------------------------------------
# [Step 3] 10-Fold Loop 및 학습
# -----------------------------------------------------------------------------
fold_chunks = np.array_split(all_indices, NUM_FOLDS)

history = []

print(f"\n>>> {NUM_FOLDS}-Fold 학습 시작 (Max Epochs: {MAX_EPOCHS})")

for fold in range(NUM_FOLDS):
    # -------------------------------------------------------------------------
    # 3-1. 데이터 분할 로직
    # -------------------------------------------------------------------------
    test_idx = fold_chunks[fold]
    val_idx = fold_chunks[(fold + 1) % NUM_FOLDS]
    
    train_chunks = []
    for i in range(NUM_FOLDS):
        if i != fold and i != (fold + 1) % NUM_FOLDS:
            train_chunks.append(fold_chunks[i])
    train_idx = np.concatenate(train_chunks)

    # Tensor 변환
    X_train = torch.tensor(X_numpy[train_idx])
    y_train = torch.tensor(y_numpy[train_idx])
    X_val = torch.tensor(X_numpy[val_idx])
    y_val = torch.tensor(y_numpy[val_idx])
    X_test = torch.tensor(X_numpy[test_idx])

    # 모델 초기화 (Fold마다 새로 생성)
    model = SimpleMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Best model tracking
    best_epoch_info = None

    print(f"\n[Fold {fold+1}/{NUM_FOLDS}] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- Epoch Loop ---
    for epoch in range(1, MAX_EPOCHS + 1):
        # ---------------------
        # [Training Phase]
        # ---------------------
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        
        train_probs_list = []
        train_labels_list = []
        
        for i in range(0, X_train.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_probs_list.append(outputs.detach().cpu().numpy())
            train_labels_list.append(batch_y.detach().cpu().numpy())

        # Train Metrics
        avg_train_loss = epoch_loss / (len(X_train) // BATCH_SIZE) if len(X_train) >= BATCH_SIZE else epoch_loss
        train_probs_concat = np.concatenate(train_probs_list)
        train_labels_concat = np.concatenate(train_labels_list)
        
        train_auc = roc_auc_score(train_labels_concat, train_probs_concat)
        train_preds = (train_probs_concat >= THRESHOLD).astype(int)
        train_acc = accuracy_score(train_labels_concat, train_preds)
        tr_prec, tr_rec, _, _ = precision_recall_fscore_support(train_labels_concat, train_preds, average=None, zero_division=0)

        # ---------------------
        # [Validation Phase]
        # ---------------------
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_probs = val_outputs.cpu().numpy()
            val_labels = y_val.cpu().numpy()

            val_auc = roc_auc_score(val_labels, val_probs)
            val_preds = (val_probs >= THRESHOLD).astype(int)
            val_acc = accuracy_score(val_labels, val_preds)
            val_prec, val_rec, _, _ = precision_recall_fscore_support(val_labels, val_preds, average=None, zero_division=0)

        # ---------------------
        # [Logging]
        # ---------------------
        log_entry = {
            "fold": fold + 1,
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "train_prec0": tr_prec[0], 
            "train_rec0": tr_rec[0],
            "train_prec1": tr_prec[1], 
            "train_rec1": tr_rec[1],
            
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_prec0": val_prec[0], 
            "val_rec0": val_rec[0],
            "val_prec1": val_prec[1], 
            "val_rec1": val_rec[1]
        }
        history.append(log_entry)

        # Best Model Check (val loss 기준)
        if val_loss > 0:
            if best_epoch_info is None or val_loss < best_epoch_info['val_loss']:
                best_epoch_info = log_entry.copy()

        # Console Log (매 epoch)
        # print(f"Fold={fold+1}, Epoch [{epoch}/{MAX_EPOCHS}] "
        #       f"Loss:{avg_train_loss:.4f} | Acc:{train_acc:.4f}/{val_acc:.4f} | AUC:{val_auc:.4f} | "
        #       f"R0:{val_rec[0]:.4f} R1:{val_rec[1]:.4f}")

    # --- Fold 완료 후 Test 예측 ---
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs_fold = test_outputs.cpu().numpy()
    
    # 결과 저장 (Global Array에 인덱스 맞춰 넣기)
    oof_probs[test_idx] = test_probs_fold

    # Fold 완료 후 Best Epoch 정보 출력
    if best_epoch_info:
        print(f"  >> [Fold {fold+1} Best Result] Epoch: {best_epoch_info['epoch']}, "
              f"Val Loss: {best_epoch_info['val_loss']:.4f}, Val Acc: {best_epoch_info['val_acc']:.4f}")

# -----------------------------------------------------------------------------
# [Step 4] 결과 저장 (CSV & NPZ)
# -----------------------------------------------------------------------------
# 1. CSV 로그 저장
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
df_history = pd.DataFrame(history)

# 컬럼 순서 정리
cols = [
    "fold", "epoch", 
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
print(f"\n>>> 상세 로그 저장 완료: {LOG_PATH}")

# 2. NPZ 파일 저장 (Pred Label, Pred Prob 추가)
os.makedirs(os.path.dirname(OOF_PATH), exist_ok=True)
oof_preds = (oof_probs >= THRESHOLD).astype(np.float32)

np.savez_compressed(
    OOF_PATH,
    pred_prob=oof_probs,
    pred_label=oof_preds
)

print(f">>> OOF 결과 파일 저장 완료: {OOF_PATH}")
print(f"    pred_prob shape: {oof_probs.shape}")
print(f"    pred_label shape: {oof_preds.shape}")
print("="*50)