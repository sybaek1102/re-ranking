import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import os

# 파일 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# [수정] OOF 기반 feature 사용
INPUT_PATH = os.path.join(INPUT_DIR, "20_re-ranking_pqD_pred_resD.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "21_re-ranking_mlp_pqD_pred_resD_oof.csv")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof", "21_re-ranking_mlp_pqD_pred_resD_oof.npz")

# 하이퍼파라미터
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 100        
THRESHOLD = 0.5
NUM_FOLDS = 10

# 1. 데이터 load & 전처리
print("\n" + "="*70)
print("📂 OOF 기반 Re-ranking 모델 학습 (10-Fold)")
print("="*70)
print("\n1. 데이터 load & 전처리")

if not os.path.exists(INPUT_PATH):
    print(f"파일이 존재하지 않습니다: {INPUT_PATH}")
    exit()

data = np.load(INPUT_PATH)
dataset = data['data'] 

# Feature 개수가 32개인지 확인 (0~31: Feature, 32: Label)
X_numpy = dataset[:, :-1].astype(np.float32)
y_numpy = dataset[:, -1].astype(np.float32).reshape(-1, 1)

print(f"  - Total Feature Shape: {X_numpy.shape}")  # (10000, 32)
print(f"  - Label Shape: {y_numpy.shape}")          # (10000, 1)
print(f"  - Label Distribution - 0: {np.sum(y_numpy == 0)}, 1: {np.sum(y_numpy == 1)}")

# 2. OOF 예측 결과물 저장을 위한 배열 초기화
num_samples = len(X_numpy)
all_indices = np.arange(num_samples)
oof_probs = np.zeros((num_samples, 1), dtype=np.float32)

# 3. MLP model 설계
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 입력 차원 32 (PQ Dist 16 + OOF-based Residual Dist 16)
        self.network = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 4. main 학습
print(f"\n4. main 학습 시작: {NUM_FOLDS}-Fold, Max Epochs: {MAX_EPOCHS}")
fold_chunks = np.array_split(all_indices, NUM_FOLDS)
history = []

for fold in range(NUM_FOLDS):
    # 4-1. fold 에 맞게 test/val/train 인덱스 세팅
    test_idx = fold_chunks[fold]
    val_idx = fold_chunks[(fold + 1) % NUM_FOLDS]
    
    train_chunks = []
    for i in range(NUM_FOLDS):
        if i != fold and i != (fold + 1) % NUM_FOLDS:
            train_chunks.append(fold_chunks[i])
    train_idx = np.concatenate(train_chunks)

    # [핵심 수정] Fold별 독립적인 Split Scaling 적용
    # -------------------------------------------------------------
    # 1) Raw 데이터 슬라이싱
    X_train_raw = X_numpy[train_idx]
    X_val_raw = X_numpy[val_idx]
    X_test_raw = X_numpy[test_idx]

    # 2) Feature 그룹 분리 (앞 16개: PQ Dist / 뒤 16개: OOF-based Residual Dist)
    X_train_f1, X_train_f2 = X_train_raw[:, :16], X_train_raw[:, 16:]
    X_val_f1, X_val_f2 = X_val_raw[:, :16], X_val_raw[:, 16:]
    X_test_f1, X_test_f2 = X_test_raw[:, :16], X_test_raw[:, 16:]

    # 3) Scaler 정의 및 Train 데이터로 Fit
    scaler_f1 = StandardScaler()
    scaler_f2 = StandardScaler()

    X_train_f1_scaled = scaler_f1.fit_transform(X_train_f1)
    X_train_f2_scaled = scaler_f2.fit_transform(X_train_f2)

    # 4) Val, Test 데이터는 Train 기준 Scaler로 Transform만 수행
    X_val_f1_scaled = scaler_f1.transform(X_val_f1)
    X_val_f2_scaled = scaler_f2.transform(X_val_f2)

    X_test_f1_scaled = scaler_f1.transform(X_test_f1)
    X_test_f2_scaled = scaler_f2.transform(X_test_f2)

    # 5) 다시 병합 (Concatenate)
    X_train_scaled = np.hstack([X_train_f1_scaled, X_train_f2_scaled])
    X_val_scaled = np.hstack([X_val_f1_scaled, X_val_f2_scaled])
    X_test_scaled = np.hstack([X_test_f1_scaled, X_test_f2_scaled])
    # -------------------------------------------------------------

    # 텐서 변환
    X_train = torch.tensor(X_train_scaled)
    y_train = torch.tensor(y_numpy[train_idx])
    X_val = torch.tensor(X_val_scaled)
    y_val = torch.tensor(y_numpy[val_idx])
    X_test = torch.tensor(X_test_scaled)

    # 4-2. 모델 초기화 (Fold마다 새로 생성)
    model = SimpleMLP()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_epoch_info = None
    print(f"\n[Fold {fold+1}/{NUM_FOLDS}] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        train_probs_list, train_labels_list = [], []
        
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

        # Metric 계산 (Train)
        avg_train_loss = epoch_loss / max(1, (len(X_train) // BATCH_SIZE))
        train_probs_concat = np.concatenate(train_probs_list)
        train_labels_concat = np.concatenate(train_labels_list)
        train_auc = roc_auc_score(train_labels_concat, train_probs_concat)
        train_preds = (train_probs_concat >= THRESHOLD).astype(int)
        train_acc = accuracy_score(train_labels_concat, train_preds)
        tr_prec, tr_rec, _, _ = precision_recall_fscore_support(train_labels_concat, train_preds, average=None, zero_division=0)

        # Validation
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

        # Log entry 생성
        log_entry = {
            "fold": fold + 1, "epoch": epoch, "train_loss": avg_train_loss, "train_acc": train_acc, "train_auc": train_auc,
            "train_prec0": tr_prec[0], "train_rec0": tr_rec[0], "train_prec1": tr_prec[1], "train_rec1": tr_rec[1],
            "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc,
            "val_prec0": val_prec[0], "val_rec0": val_rec[0], "val_prec1": val_prec[1], "val_rec1": val_rec[1]
        }
        history.append(log_entry)

        # Best epoch 체크
        if val_loss > 0:
            if best_epoch_info is None or val_loss < best_epoch_info['val_loss']:
                best_epoch_info = log_entry.copy()

    # Fold 종료 후 Test(OOF) 예측
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        oof_probs[test_idx] = test_outputs.cpu().numpy()

    if best_epoch_info:
        print(f"  >> [Fold {fold+1} Best] Epoch: {best_epoch_info['epoch']}, "
              f"Val Loss: {best_epoch_info['val_loss']:.4f}, Val AUC: {best_epoch_info['val_auc']:.4f}, "
              f"Val Acc: {best_epoch_info['val_acc']:.4f}")

# 5. 결과 저장
print("\n" + "="*70)
print("5. 결과 저장")
print("="*70)

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
df_history = pd.DataFrame(history)
df_history.to_csv(LOG_PATH, index=False)
print(f"✓ 상세 로그 저장 완료: {LOG_PATH}")

os.makedirs(os.path.dirname(OOF_PATH), exist_ok=True)
oof_preds = (oof_probs >= THRESHOLD).astype(np.float32)
np.savez_compressed(OOF_PATH, pred_prob=oof_probs, pred_label=oof_preds)

print(f"✓ OOF 결과 파일 저장 완료: {OOF_PATH}")
print(f"  - pred_prob shape: {oof_probs.shape}")
print(f"  - pred_label shape: {oof_preds.shape}")

# 6. 전체 OOF 성능 평가
print("\n" + "="*70)
print("6. 전체 OOF 성능")
print("="*70)

oof_auc = roc_auc_score(y_numpy, oof_probs)
oof_pred_labels = (oof_probs >= THRESHOLD).astype(int)
oof_acc = accuracy_score(y_numpy, oof_pred_labels)
oof_prec, oof_rec, _, _ = precision_recall_fscore_support(y_numpy, oof_pred_labels, average=None, zero_division=0)

print(f"  - Overall OOF Accuracy:  {oof_acc:.4f}")
print(f"  - Overall OOF AUC:       {oof_auc:.4f}")
print(f"  - Class 0 Precision:     {oof_prec[0]:.4f}")
print(f"  - Class 0 Recall:        {oof_rec[0]:.4f}")
print(f"  - Class 1 Precision:     {oof_prec[1]:.4f}")
print(f"  - Class 1 Recall:        {oof_rec[1]:.4f}")

print("\n" + "="*70)
print("[Feature 정보]")
print("  - PQ Distance Features:        16 dims (앞 16개)")
print("  - OOF-based Residual Distance: 16 dims (||R||² - 2*pred(⟨Q-C,R⟩))")
print("  - Total:                       32 dims")
print("="*70)

print("\n✅ 모든 작업 완료!")