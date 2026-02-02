import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import sys

# =====================================================================
# 파일 경로 설정
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

FEATURE_PATH = os.path.join(INPUT_DIR, "11_residual_features.npz")
LABEL_PATH = os.path.join(INPUT_DIR, "11_residual_label.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "11_residual_mlp.csv")

# =====================================================================
# 하이퍼파라미터
# =====================================================================
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
EPOCHS = 100
VAL_RATIO = 0.2
PATIENCE = 15  # Early stopping

# 모델 구조
FEATURE_DIM = 14        # 각 subspace feature 차원
SHARED_HIDDEN = 32      # Shared MLP 중간 차원
EMBED_DIM = 8           # Shared MLP 출력 차원
GLOBAL_HIDDEN = 64      # Global MLP 중간 차원

# Multi-Task Loss 가중치
ALPHA = 0.1   # Local loss weight (보조)
BETA = 0.9    # Global loss weight (주요)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")
print(f"📊 Multi-Task Weights: Alpha={ALPHA}, Beta={BETA}")

# =====================================================================
# 1. 데이터 로드 & 전처리
# =====================================================================
print("\n" + "="*70)
print("1️⃣  데이터 로드 & 전처리")
print("="*70)

if not os.path.exists(FEATURE_PATH) or not os.path.exists(LABEL_PATH):
    print(f"❌ Error: 파일을 찾을 수 없습니다.")
    print(f"   Feature: {FEATURE_PATH}")
    print(f"   Label: {LABEL_PATH}")
    sys.exit()

# 데이터 로드
X_np = np.load(FEATURE_PATH)["data"].astype(np.float32)  # (160000, 16, 14)
y_np = np.load(LABEL_PATH)["data"].astype(np.float32)    # (160000, 16, 1)

print(f"✓ Feature Shape: {X_np.shape}")
print(f"✓ Label Shape: {y_np.shape}")

# Local labels: 각 subspace별 값
y_local = y_np  # (160000, 16, 1)

# Global labels: 16개 subspace의 합
y_global = np.sum(y_np, axis=1)  # (160000, 1)

print(f"✓ Local Label Shape: {y_local.shape}")
print(f"✓ Global Label Shape: {y_global.shape}")

# 통계 확인
print(f"\n📊 Global Label Statistics:")
print(f"   Mean: {y_global.mean():.2f}")
print(f"   Std:  {y_global.std():.2f}")
print(f"   Min:  {y_global.min():.2f}")
print(f"   Max:  {y_global.max():.2f}")

# Target 정규화 (Z-score normalization)
y_global_mean = y_global.mean()
y_global_std = y_global.std()
y_global_normalized = (y_global - y_global_mean) / y_global_std

y_local_mean = y_local.mean()
y_local_std = y_local.std()
y_local_normalized = (y_local - y_local_mean) / y_local_std

print(f"\n✓ Normalization Applied:")
print(f"   Global: mean={y_global_mean:.2f}, std={y_global_std:.2f}")
print(f"   Local: mean={y_local_mean:.2f}, std={y_local_std:.2f}")

# =====================================================================
# 2. Train & Val Split
# =====================================================================
print("\n" + "="*70)
print("2️⃣  Train & Validation Split")
print("="*70)

indices = np.arange(len(X_np))
train_idx, val_idx = train_test_split(
    indices, test_size=VAL_RATIO, random_state=42
)

X_train = X_np[train_idx]
X_val = X_np[val_idx]
y_local_train = y_local_normalized[train_idx]
y_local_val = y_local_normalized[val_idx]
y_global_train = y_global_normalized[train_idx]
y_global_val = y_global_normalized[val_idx]

# 원본 값도 저장 (evaluation용)
y_global_val_original = y_global[val_idx]

print(f"✓ Train Samples: {len(train_idx)}")
print(f"✓ Val Samples: {len(val_idx)}")

# DataLoader 생성
train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_local_train),
    torch.tensor(y_global_train)
)
val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_local_val),
    torch.tensor(y_global_val)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True)

# =====================================================================
# 3. Multi-Task Model 설계
# =====================================================================
print("\n" + "="*70)
print("3️⃣  Multi-Task Model 설계")
print("="*70)

class MultiTaskDistancePredictor(nn.Module):
    def __init__(self):
        super(MultiTaskDistancePredictor, self).__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(FEATURE_DIM)
        
        # Shared MLP: (14) → (32) → (8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, SHARED_HIDDEN),
            nn.LayerNorm(SHARED_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(SHARED_HIDDEN, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.LeakyReLU(0.1)
        )
        
        # Local Head: 각 subspace 거리 예측
        self.local_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 1)
        )
        
        # Global Head: 전체 거리 예측
        global_input_dim = 16 * EMBED_DIM  # 128
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, GLOBAL_HIDDEN),
            nn.LayerNorm(GLOBAL_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(GLOBAL_HIDDEN, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, 16, 14)
        batch_size = x.size(0)
        
        # Flatten for shared processing
        x_flat = x.view(-1, FEATURE_DIM)  # (batch*16, 14)
        
        # Input normalization
        x_norm = self.input_norm(x_flat)
        
        # Shared encoding
        embeddings = self.shared_mlp(x_norm)  # (batch*16, 8)
        
        # Local predictions (각 subspace별)
        local_preds = self.local_head(embeddings)  # (batch*16, 1)
        local_preds = local_preds.view(batch_size, 16, 1)
        
        # Global prediction (전체)
        global_input = embeddings.view(batch_size, -1)  # (batch, 128)
        global_pred = self.global_mlp(global_input)  # (batch, 1)
        
        return local_preds, global_pred

# 모델 초기화
model = MultiTaskDistancePredictor().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

print(model)
print(f"\n✓ Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

# =====================================================================
# 4. Metric 계산 함수
# =====================================================================
def calculate_metrics(y_true, y_pred):
    """
    Regression metrics 계산
    y_true, y_pred: numpy arrays (denormalized)
    """
    # MSE, MAE, RMSE
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # R² Score
    r2 = r2_score(y_true, y_pred)
    
    # Correlation
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # Normalized RMSE (0~1 범위)
    y_range = y_true.max() - y_true.min()
    nrmse = 1 - (rmse / y_range) if y_range > 0 else 0
    
    # MAPE Score (0~1 범위)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    mape_score = 1 / (1 + mape / 100)
    
    # Tolerance-based Accuracy (표준편차의 10% 내)
    tolerance = y_true.std() * 0.1
    acc_like = np.mean(np.abs(y_true - y_pred) < tolerance)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'corr': corr,
        'nrmse': nrmse,
        'mape_score': mape_score,
        'acc_like': acc_like
    }

# =====================================================================
# 5. 학습 루프
# =====================================================================
print("\n" + "="*70)
print("4️⃣  학습 시작")
print("="*70)

history = []
best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    # ==================== Train ====================
    model.train()
    train_loss_local_sum = 0
    train_loss_global_sum = 0
    train_loss_total_sum = 0
    
    for batch_X, batch_y_local, batch_y_global in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y_local = batch_y_local.to(DEVICE)
        batch_y_global = batch_y_global.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward
        local_preds, global_pred = model(batch_X)
        
        # Loss 계산
        loss_local = nn.MSELoss()(local_preds, batch_y_local)
        loss_global = nn.MSELoss()(global_pred, batch_y_global)
        loss_total = ALPHA * loss_local + BETA * loss_global
        
        # Backward
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss_local_sum += loss_local.item()
        train_loss_global_sum += loss_global.item()
        train_loss_total_sum += loss_total.item()
    
    avg_train_loss_local = train_loss_local_sum / len(train_loader)
    avg_train_loss_global = train_loss_global_sum / len(train_loader)
    avg_train_loss_total = train_loss_total_sum / len(train_loader)
    
    # ==================== Validation ====================
    model.eval()
    val_loss_local_sum = 0
    val_loss_global_sum = 0
    val_loss_total_sum = 0
    
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y_local, batch_y_global in val_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y_local = batch_y_local.to(DEVICE)
            batch_y_global = batch_y_global.to(DEVICE)
            
            # Forward
            local_preds, global_pred = model(batch_X)
            
            # Loss
            loss_local = nn.MSELoss()(local_preds, batch_y_local)
            loss_global = nn.MSELoss()(global_pred, batch_y_global)
            loss_total = ALPHA * loss_local + BETA * loss_global
            
            val_loss_local_sum += loss_local.item()
            val_loss_global_sum += loss_global.item()
            val_loss_total_sum += loss_total.item()
            
            # Denormalize for metrics
            global_pred_denorm = global_pred.cpu().numpy() * y_global_std + y_global_mean
            all_val_preds.append(global_pred_denorm)
            
        # Concatenate predictions
        all_val_labels = y_global_val_original
        all_val_preds = np.concatenate(all_val_preds)
    
    avg_val_loss_local = val_loss_local_sum / len(val_loader)
    avg_val_loss_global = val_loss_global_sum / len(val_loader)
    avg_val_loss_total = val_loss_total_sum / len(val_loader)
    
    # Metrics 계산
    val_metrics = calculate_metrics(all_val_labels, all_val_preds)
    
    # Learning rate scheduler
    scheduler.step(avg_val_loss_total)
    
    # Log entry
    log_entry = {
        'epoch': epoch,
        'train_loss_local': avg_train_loss_local,
        'train_loss_global': avg_train_loss_global,
        'train_loss_total': avg_train_loss_total,
        'val_loss_local': avg_val_loss_local,
        'val_loss_global': avg_val_loss_global,
        'val_loss_total': avg_val_loss_total,
        'val_mse': val_metrics['mse'],
        'val_mae': val_metrics['mae'],
        'val_rmse': val_metrics['rmse'],
        'val_r2': val_metrics['r2'],
        'val_corr': val_metrics['corr'],
        'val_nrmse': val_metrics['nrmse'],
        'val_mape_score': val_metrics['mape_score'],
        'val_acc_like': val_metrics['acc_like'],
        'lr': optimizer.param_groups[0]['lr']
    }
    history.append(log_entry)
    
    # Best model 체크
    if avg_val_loss_total < best_val_loss:
        best_val_loss = avg_val_loss_total
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Console 출력
    print(f"Epoch [{epoch:3d}/{EPOCHS}] "
          f"Loss: {avg_train_loss_total:.4f}/{avg_val_loss_total:.4f} | "
          f"R²: {val_metrics['r2']:.4f} | "
          f"Corr: {val_metrics['corr']:.4f} | "
          f"NRMSE: {val_metrics['nrmse']:.4f} | "
          f"AccLike: {val_metrics['acc_like']:.4f}")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⚠️  Early stopping at epoch {epoch} (patience={PATIENCE})")
        break

# =====================================================================
# 6. 결과 저장 및 출력
# =====================================================================
print("\n" + "="*70)
print("5️⃣  결과 저장 및 출력")
print("="*70)

# CSV 저장
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
df_history = pd.DataFrame(history)
df_history.to_csv(LOG_PATH, index=False)
print(f"✓ 학습 로그 저장: {LOG_PATH}")

# Best epoch 정보 출력
best_log = df_history.iloc[best_epoch - 1]
print(f"\n{'='*70}")
print(f"🏆 Best Performance @ Epoch {best_epoch}")
print(f"{'='*70}")
print(f"  Total Loss:      {best_log['val_loss_total']:.4f}")
print(f"  Global Loss:     {best_log['val_loss_global']:.4f}")
print(f"  Local Loss:      {best_log['val_loss_local']:.4f}")
print(f"{'-'*70}")
print(f"  MSE:             {best_log['val_mse']:.2f}")
print(f"  MAE:             {best_log['val_mae']:.2f}")
print(f"  RMSE:            {best_log['val_rmse']:.2f}")
print(f"  R² Score:        {best_log['val_r2']:.4f}")
print(f"  Correlation:     {best_log['val_corr']:.4f}")
print(f"  NRMSE:           {best_log['val_nrmse']:.4f}")
print(f"  MAPE Score:      {best_log['val_mape_score']:.4f}")
print(f"  Acc-like (10%):  {best_log['val_acc_like']:.4f}")
print(f"{'='*70}")

print("\n✅ 학습 완료!")