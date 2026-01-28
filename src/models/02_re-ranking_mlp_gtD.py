import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler # 추가
import os

# 파일 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# 입력 파일 및 로그 파일 경로 수정
INPUT_PATH = os.path.join(INPUT_DIR, "02_re-ranking_features_gtD.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "02_re-ranking_mlp_gtD.csv")

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

X_numpy = dataset[:, :-1].astype(np.float32)
y_numpy = dataset[:, -1].astype(np.float32).reshape(-1, 1)

print(f"  - Raw Feature Mean: {np.mean(X_numpy):.4f}, Std: {np.std(X_numpy):.4f}")

# 2. train & val split 및 Scaling
print("\n2. train & val split & Scaling")
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_numpy, y_numpy, test_size=VAL_RATIO, random_state=42, stratify=y_numpy
)

# [핵심 수정] StandardScaler 적용
# Train 데이터를 기준으로 학습(fit)하고, Val 데이터는 train 기준에 맞춰 변환(transform)만 수행
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

print(f"  - Scaled Train Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
print(f"  - Train Samples: {len(X_train_tensor)}")
print(f"  - Val Samples: {len(X_val_tensor)}")

# 3. MLP model 설계 (구조 유지)
print("\n3. MLP model 설계")
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

print("\n" + "="*50)
print(f">>> 학습 종료. 상세 로그 저장 완료: {LOG_PATH}")
print("="*50)

# 6. 최종 확인
print("\n[Best Validation 성능 (Loss 기준)]")
print(f"Epoch    : {best_epoch_info['epoch']}")
print(f"Accuracy : {best_epoch_info['val_acc']:.4f}")
print(f"AUC      : {best_epoch_info['val_auc']:.4f}")
print(f"Class 1 Recall (Re-ranking target): {best_epoch_info['val_rec1']:.4f}")
print("="*50)