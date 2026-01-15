import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 정규화를 위해 추가
import os
import sys

# 파일 경로 설정 (상대 경로 사용)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

FEATURE_PATH = os.path.join(INPUT_DIR, "residual-sign_features.npz")
LABEL_PATH = os.path.join(INPUT_DIR, "residual-sign_label.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "residual-sign_mlp_add_id_feature.csv")

# 하이퍼파라미터
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
EPOCHS = 100
VAL_RATIO = 0.2
THRESHOLD = 0.5 
FEATURE_DIM = 19            # [수정됨] 18(기존) + 1(ID) = 19차원
EMBED_DIM = 8               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 데이터 load & 전처리
print("\n1. 데이터 load & 전처리")
if not os.path.exists(FEATURE_PATH) or not os.path.exists(LABEL_PATH):
    print(f"Error: 파일을 찾을 수 없습니다.")
    print(f"Feature: {FEATURE_PATH}")
    print(f"Label: {LABEL_PATH}")
    sys.exit()

X_np = np.load(FEATURE_PATH)["data"].astype(np.float32) 
y_np = np.load(LABEL_PATH)["data"].astype(np.float32)   

print(f"  - (Original) Feature Shape: {X_np.shape}")     # (160000, 16, 18)

# id feature 생성 
N, S, D = X_np.shape # N=160000, S=16, D=18

# 1) ID 생성 (1 ~ 16)
ids = np.arange(1, 17).reshape(-1, 1).astype(np.float32) # (16,) -> (16, 1)

# 2) ID 정규화 (StandardScaler)
scaler = StandardScaler()
ids_scaled = scaler.fit_transform(ids) 

# 3) 전체 데이터에 맞게 확장
ids_expanded = np.tile(ids_scaled, (N, 1, 1)) # (16, 1) -> (160000, 16, 1)

# 4) 기존 Feature에 합치기
X_np = np.concatenate([X_np, ids_expanded], axis=2) # (160000, 16, 18) + (160000, 16, 1)

print(f"  - (Modified) Feature Shape: {X_np.shape}")     # (160000, 16, 19)
print(f"  - Label Shape: {y_np.shape}")       # (160000, 1)

X_tensor = torch.tensor(X_np)
y_tensor = torch.tensor(y_np)

# 2. train & val split
print("\n2. train & val split")

X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, 
    test_size=VAL_RATIO, 
    random_state=42, 
    stratify=y_np       
)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
print(f"  - Train Samples: {len(train_dataset)}")
print(f"  - Val Samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 3. MLP model 설계
class ResidualSignPredictionModel(nn.Module):
    def __init__(self):
        super(ResidualSignPredictionModel, self).__init__()
        self.input_norm = nn.BatchNorm1d(FEATURE_DIM)               
        
        # MLP1: feature(19D) -> 32D -> 8D 
        self.shared_mlp = nn.Sequential(                            
            nn.Linear(FEATURE_DIM, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, EMBED_DIM),
            nn.LeakyReLU() 
        )
        
        # MLP2(residual 부호 예측): 8D*16개(128D) -> 64D -> 1D
        global_input_dim = 16 * EMBED_DIM
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # (Batch, 16, 19) -> (Batch * 16, 19)로 펼침
        x_flat = x.view(-1, FEATURE_DIM)
        
        # 스케일링
        x_norm = self.input_norm(x_flat)
        
        # MLP1 (19D -> 32D -> 8D)
        block_emb = self.shared_mlp(x_norm)
        
        # MLP2 (8D*16개(128D) -> 64D -> 1D)
        global_input = block_emb.view(batch_size, -1)
        return self.global_mlp(global_input)

# 모델 초기화
model = ResidualSignPredictionModel().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n3. MLP model 설계")
print(model)

# metric 측정 함수
def calculate_metrics(y_true, y_pred_prob):
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)
    
    acc = accuracy_score(y_true, y_pred)  
    try:                                            
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = 0.0 

    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    if len(prec) == 2: 
        p0, p1 = prec[0], prec[1]
        r0, r1 = rec[0], rec[1]
    else:
        p0, p1 = 0.0, 0.0
        r0, r1 = 0.0, 0.0
        
    return acc, auc, p0, p1, r0, r1


# 4. main
print(f"\n4. main 학습 시작 (Total Epochs: {EPOCHS})")
history = []
best_val_acc = 0.0
best_epoch_info = None

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    train_loss_sum = 0
    all_train_labels = []
    all_train_probs = []
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        
        loss_fn = nn.BCELoss() 
        loss = loss_fn(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        train_loss_sum += loss.item()
        
        all_train_probs.append(outputs.detach().cpu().numpy())
        all_train_labels.append(batch_y.detach().cpu().numpy())
        
    # train summary
    train_probs = np.concatenate(all_train_probs)
    train_labels = np.concatenate(all_train_labels)
    avg_train_loss = train_loss_sum / len(train_loader)
    
    t_acc, t_auc, t_p0, t_p1, t_r0, t_r1 = calculate_metrics(train_labels, train_probs)
    
    # validation
    model.eval()
    val_loss_sum = 0
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            
            loss_fn = nn.BCELoss()
            loss = loss_fn(outputs, batch_y)
            
            val_loss_sum += loss.item()
            
            all_val_probs.append(outputs.cpu().numpy())
            all_val_labels.append(batch_y.cpu().numpy())
            
    # validation summary
    val_probs = np.concatenate(all_val_probs)
    val_labels = np.concatenate(all_val_labels)
    avg_val_loss = val_loss_sum / len(val_loader)
    
    v_acc, v_auc, v_p0, v_p1, v_r0, v_r1 = calculate_metrics(val_labels, val_probs)
    
    # Log
    log_entry = {
        "epoch": epoch,
        "train_loss": avg_train_loss, "val_loss": avg_val_loss,
        "train_acc": t_acc, "val_acc": v_acc,
        "train_auc": t_auc, "val_auc": v_auc,
        "train_prec0": t_p0, "val_prec0": v_p0,
        "train_prec1": t_p1, "val_prec1": v_p1,
        "train_rec0": t_r0, "val_rec0": v_r0,
        "train_rec1": t_r1, "val_rec1": v_r1
    }
    history.append(log_entry)
    
    # Best Model Check (val loss 기준)
    if v_auc > 0:
        if best_epoch_info is None or avg_val_loss < best_epoch_info['val_loss']:
            best_val_acc = v_acc
            best_epoch_info = log_entry
    
    # Console Log
    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"Loss:{avg_train_loss:.4f} | Acc:{t_acc:.4f}/{v_acc:.4f} | AUC:{v_auc:.4f} | "
          f"R0:{v_r0:.2f} R1:{v_r1:.2f}")


# 5. metric 결과 출력 및 csv 파일에 저장
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

df_history = pd.DataFrame(history)
df_history.to_csv(LOG_PATH, index=False)
print(f"\n5. metric 결과 csv 파일에 저장 완료: {LOG_PATH}")

if best_epoch_info:
    print(f"[Best Performance @ Epoch {best_epoch_info['epoch']}]")
    print(f"    - Loss      : {best_epoch_info['val_loss']:.4f}")
    print(f"    - Accuracy  : {best_epoch_info['val_acc']:.4f}")
    print(f"    - ROC-AUC   : {best_epoch_info['val_auc']:.4f}")
    print("-" * 40)
    print(f"    - Class 0 (Negative) | Precision: {best_epoch_info['val_prec0']:.4f}, Recall: {best_epoch_info['val_rec0']:.4f}")
    print(f"    - Class 1 (Positive) | Precision: {best_epoch_info['val_prec1']:.4f}, Recall: {best_epoch_info['val_rec1']:.4f}")
    print("="*60)
else:
    print("학습 기록이 없습니다.")