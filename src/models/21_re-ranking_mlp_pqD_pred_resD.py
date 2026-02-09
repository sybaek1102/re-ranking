import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import os

# 파일 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# [수정] OOF 기반 feature 사용
INPUT_PATH = os.path.join(INPUT_DIR, "22_re-ranking_pqD_pred_resD_int4_norm_scaled.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "22_re-ranking_pqD_pred_resD_int4_norm_scaled.csv")

# 하이퍼파라미터
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 100
VAL_RATIO = 0.2
THRESHOLD = 0.5

# 1. 데이터 load & 전처리
print("\n1. 데이터 load & 전처리")
if not os.path.exists(INPUT_PATH):
    print(f"파일이 존재하지 않습니다: {INPUT_PATH}")
    exit()

data = np.load(INPUT_PATH)
dataset = data['data'] 

# Feature 개수가 32개인지 확인 (0~31: Feature, 32: Label)
X_numpy = dataset[:, :-1].astype(np.float32)
y_numpy = dataset[:, -1].astype(np.float32).reshape(-1, 1)

print(f"  - Total Feature Shape: {X_numpy.shape}") # (10000, 32)
print(f"  - Label Shape: {y_numpy.shape}") # (10000, 1)
print(f"  - Raw Feature Mean: {np.mean(X_numpy):.4f}, Std: {np.std(X_numpy):.4f}")

# Label 분포 확인
print(f"  - Label Distribution - 0: {np.sum(y_numpy == 0)}, 1: {np.sum(y_numpy == 1)}")

# 2. train & val split 및 Scaling
print("\n2. train & val split & Scaling")
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_numpy, y_numpy, test_size=VAL_RATIO, random_state=42, stratify=y_numpy
)

# [핵심 수정] 분리형 StandardScaler 적용
print("  >>> Feature 그룹별 개별 Scaling 적용 중...")

# 그룹 분리 (앞 16개: PQ Dist / 뒤 16개: OOF-based Residual Dist)
X_train_f1 = X_train_raw[:, :16]
X_train_f2 = X_train_raw[:, 16:]

X_val_f1 = X_val_raw[:, :16]
X_val_f2 = X_val_raw[:, 16:]

# 각각의 Scaler 정의
scaler_f1 = StandardScaler()
scaler_f2 = StandardScaler()

# 각각 Fit & Transform (Train 기준)
X_train_f1_scaled = scaler_f1.fit_transform(X_train_f1)
X_train_f2_scaled = scaler_f2.fit_transform(X_train_f2)

# Transform (Validation은 Train 기준으로 변환만)
X_val_f1_scaled = scaler_f1.transform(X_val_f1)
X_val_f2_scaled = scaler_f2.transform(X_val_f2)

# 다시 병합 (Concatenate)
X_train = np.hstack([X_train_f1_scaled, X_train_f2_scaled])
X_val = np.hstack([X_val_f1_scaled, X_val_f2_scaled])

# Tensor 변환
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

print(f"  - Scaled Train Shape: {X_train.shape}")
print(f"  - Train Samples: {len(X_train_tensor)}")
print(f"  - Val Samples: {len(X_val_tensor)}")

# 3. MLP model 설계
print("\n3. MLP model 설계")
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

model = SimpleMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"  - Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 4. main 학습 시작
print(f"\n4. main 학습 시작 (Total Epochs: {EPOCHS})")
history = [] 
best_epoch_info = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0
    
    train_probs_list, train_labels_list = [], []
    
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

    # Train Metric 계산
    train_probs_concat = np.concatenate(train_probs_list)
    train_labels_concat = np.concatenate(train_labels_list)
    avg_t_loss = epoch_loss / max(1, (len(X_train_tensor) // BATCH_SIZE))
    t_auc = roc_auc_score(train_labels_concat, train_probs_concat)
    train_preds = (train_probs_concat >= THRESHOLD).astype(int)
    t_acc = accuracy_score(train_labels_concat, train_preds)
    t_prec, t_rec, _, _ = precision_recall_fscore_support(train_labels_concat, train_preds, average=None, zero_division=0)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        v_loss = criterion(val_outputs, y_val_tensor).item()
        val_probs = val_outputs.cpu().numpy()
        val_labels = y_val_tensor.cpu().numpy()
        
        v_auc = roc_auc_score(val_labels, val_probs)
        val_preds = (val_probs >= THRESHOLD).astype(int)
        v_acc = accuracy_score(val_labels, val_preds)
        v_prec, v_rec, _, _ = precision_recall_fscore_support(val_labels, val_preds, average=None, zero_division=0)

    # Log 저장
    log_entry = {
        "epoch": epoch, "train_loss": avg_t_loss, "val_loss": v_loss,
        "train_acc": t_acc, "val_acc": v_acc, "train_auc": t_auc, "val_auc": v_auc,
        "train_prec0": t_prec[0], "val_prec0": v_prec[0], "train_prec1": t_prec[1], "val_prec1": v_prec[1],
        "train_rec0": t_rec[0], "val_rec0": v_rec[0], "train_rec1": t_rec[1], "val_rec1": v_rec[1],
    }
    history.append(log_entry)

    if best_epoch_info is None or v_loss < best_epoch_info['val_loss']:
        best_epoch_info = log_entry
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{EPOCHS}] Loss:{avg_t_loss:.4f} | V_Loss:{v_loss:.4f} | V_Acc:{v_acc:.4f} | V_AUC:{v_auc:.4f}")

# 5. 결과 저장
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
df_history = pd.DataFrame(history)
df_history.to_csv(LOG_PATH, index=False)

print("\n" + "="*70)
print(f">>> 학습 종료. 상세 로그 저장 완료: {LOG_PATH}")
print("="*70)

# 6. 최종 확인
print("\n[Best Validation 성능 (Loss 기준)]")
print(f"Epoch    : {best_epoch_info['epoch']}")
print(f"Accuracy : {best_epoch_info['val_acc']:.4f}")
print(f"AUC      : {best_epoch_info['val_auc']:.4f}")
print(f"Precision (Class 1): {best_epoch_info['val_prec1']:.4f}")
print(f"Recall (Class 1):    {best_epoch_info['val_rec1']:.4f}")
print("="*70)

print("\n[Feature 정보]")
print("  - PQ Distance Features:        16 dims (앞 16개)")
print("  - OOF-based Residual Distance: 16 dims (||R||² - 2*pred(⟨Q-C,R⟩))")
print("  - Total:                       32 dims")
print("="*70)