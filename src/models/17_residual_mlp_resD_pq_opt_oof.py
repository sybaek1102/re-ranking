import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# =====================================================================
# 파일 경로 설정
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

FEATURE_PATH = os.path.join(INPUT_DIR, "17_residual_features_resD_pq_opt.npz")
LABEL_PATH = os.path.join(INPUT_DIR, "15_residual_label_dot.npz")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "17_residual_mlp_resD_pq_opt_dot_oof.csv")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof", "17_residual_mlp_resD_pq_opt_dot_oof.npz")

# =====================================================================
# 하이퍼파라미터
# =====================================================================
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_FOLDS = 10  # 10-Fold OOF

# 모델 구조
FEATURE_DIM = 10        # 각 subspace feature 차원 (10 dims)
SHARED_HIDDEN = 32      # Shared MLP 중간 차원
EMBED_DIM = 8           # Shared MLP 출력 차원
GLOBAL_HIDDEN = 64      # Global MLP 중간 차원

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")
print(f"📁 {NUM_FOLDS}-Fold OOF Training")
print(f"📊 Feature Dimension: {FEATURE_DIM}")

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
X_np = np.load(FEATURE_PATH)["data"].astype(np.float32)  # (160000, 16, 10)
y_np = np.load(LABEL_PATH)["data"].astype(np.float32)    # (160000, 16, 1)

print(f"✓ Feature Shape: {X_np.shape}")
print(f"✓ Label Shape: {y_np.shape}")

# Feature 구성 확인
print(f"\n📋 Feature 구성 (10 dims per subspace):")
print(f"   product_vec(8) | feat_res_dot(1) | feat_res_norm_sc(1)")
print(f"   Removed: distance_l2, centroid_mae, centroid_var, feat_res_sq")

# Global labels: 16개 subspace의 합
y_global = np.sum(y_np, axis=1)  # (160000, 1)

print(f"\n✓ Global Label Shape: {y_global.shape}")

# 통계 확인
print(f"\n📊 Global Label Statistics:")
print(f"   Mean: {y_global.mean():.2f}")
print(f"   Std:  {y_global.std():.2f}")
print(f"   Min:  {y_global.min():.2f}")
print(f"   Max:  {y_global.max():.2f}")

# Target 정규화를 위한 전역 통계 (전체 데이터 기준)
y_global_mean = y_global.mean()
y_global_std = y_global.std()

print(f"\n✓ Normalization Stats (전체 데이터 기준):")
print(f"   Global: mean={y_global_mean:.2f}, std={y_global_std:.2f}")

# OOF 예측 결과를 저장할 배열 초기화
num_samples = len(X_np)
oof_preds = np.zeros((num_samples, 1), dtype=np.float32)

# =====================================================================
# 2. Fold Split 생성
# =====================================================================
print("\n" + "="*70)
print("2️⃣  10-Fold Split 생성")
print("="*70)

all_indices = np.arange(num_samples)
fold_chunks = np.array_split(all_indices, NUM_FOLDS)

print(f"✓ Total Samples: {num_samples}")
print(f"✓ Samples per Fold: ~{len(fold_chunks[0])}")

# =====================================================================
# 3. Single-Task Model 정의
# =====================================================================
print("\n" + "="*70)
print("3️⃣  Single-Task Model 설계")
print("="*70)

class SingleTaskDistancePredictor(nn.Module):
    def __init__(self):
        super(SingleTaskDistancePredictor, self).__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(FEATURE_DIM)
        
        # Shared MLP: (10) → (32) → (8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, SHARED_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(SHARED_HIDDEN, EMBED_DIM),
            nn.LeakyReLU(0.1)
        )
        
        # Global MLP: 전체 거리 예측
        global_input_dim = 16 * EMBED_DIM  # 128
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, GLOBAL_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(GLOBAL_HIDDEN, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, 16, 10)
        batch_size = x.size(0)
        
        # Flatten for shared processing
        x_flat = x.view(-1, FEATURE_DIM)  # (batch*16, 10)
        
        # Input normalization
        x_norm = self.input_norm(x_flat)
        
        # Shared encoding
        embeddings = self.shared_mlp(x_norm)  # (batch*16, 8)
        
        # Global prediction (전체)
        global_input = embeddings.view(batch_size, -1)  # (batch, 128)
        global_pred = self.global_mlp(global_input)  # (batch, 1)
        
        return global_pred

# 모델 구조 출력 (한 번만)
temp_model = SingleTaskDistancePredictor()
print(temp_model)
print(f"\n✓ Total Parameters: {sum(p.numel() for p in temp_model.parameters()):,}")
del temp_model

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
    
    # Tolerance-based Accuracy (표준편차의 10%, 20%, 30%, 40%, 50% 내)
    y_std = y_true.std()
    acc_like_0_1 = np.mean(np.abs(y_true - y_pred) < y_std * 0.1)
    acc_like_0_2 = np.mean(np.abs(y_true - y_pred) < y_std * 0.2)
    acc_like_0_3 = np.mean(np.abs(y_true - y_pred) < y_std * 0.3)
    acc_like_0_4 = np.mean(np.abs(y_true - y_pred) < y_std * 0.4)
    acc_like_0_5 = np.mean(np.abs(y_true - y_pred) < y_std * 0.5)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'corr': corr,
        'nrmse': nrmse,
        'mape_score': mape_score,
        'acc_like_0.1': acc_like_0_1,
        'acc_like_0.2': acc_like_0_2,
        'acc_like_0.3': acc_like_0_3,
        'acc_like_0.4': acc_like_0_4,
        'acc_like_0.5': acc_like_0_5
    }

# =====================================================================
# 5. 10-Fold OOF 학습
# =====================================================================
print("\n" + "="*70)
print("4️⃣  10-Fold OOF 학습 시작")
print("="*70)

history = []

for fold in range(NUM_FOLDS):
    print(f"\n{'='*70}")
    print(f"📂 Fold {fold + 1}/{NUM_FOLDS}")
    print(f"{'='*70}")
    
    # ==================== Fold Split ====================
    # Test: 현재 fold
    test_idx = fold_chunks[fold]
    # Val: 다음 fold
    val_idx = fold_chunks[(fold + 1) % NUM_FOLDS]
    # Train: 나머지 fold들
    train_chunks = []
    for i in range(NUM_FOLDS):
        if i != fold and i != (fold + 1) % NUM_FOLDS:
            train_chunks.append(fold_chunks[i])
    train_idx = np.concatenate(train_chunks)
    
    print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Data 추출 및 정규화
    X_train = X_np[train_idx]
    X_val = X_np[val_idx]
    X_test = X_np[test_idx]
    
    y_global_train = (y_global[train_idx] - y_global_mean) / y_global_std
    y_global_val = (y_global[val_idx] - y_global_mean) / y_global_std
    
    # 원본 값 (evaluation용)
    y_global_val_original = y_global[val_idx]
    
    # DataLoader 생성
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_global_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_global_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # ==================== Model 초기화 (Fold마다 새로 생성) ====================
    model = SingleTaskDistancePredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    # ==================== Epoch Loop ====================
    for epoch in range(1, EPOCHS + 1):
        # ========== Train ==========
        model.train()
        train_loss_sum = 0
        
        for batch_X, batch_y_global in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y_global = batch_y_global.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            global_pred = model(batch_X)
            
            # Loss 계산
            loss = nn.MSELoss()(global_pred, batch_y_global)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # ========== Validation ==========
        model.eval()
        val_loss_sum = 0
        
        all_val_preds = []
        
        with torch.no_grad():
            for batch_X, batch_y_global in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y_global = batch_y_global.to(DEVICE)
                
                # Forward
                global_pred = model(batch_X)
                
                # Loss
                loss = nn.MSELoss()(global_pred, batch_y_global)
                
                val_loss_sum += loss.item()
                
                # Denormalize for metrics
                global_pred_denorm = global_pred.cpu().numpy() * y_global_std + y_global_mean
                all_val_preds.append(global_pred_denorm)
        
        # Concatenate predictions
        all_val_preds = np.concatenate(all_val_preds)
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # Metrics 계산
        val_metrics = calculate_metrics(y_global_val_original, all_val_preds)
        
        # Learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log entry
        log_entry = {
            'fold': fold + 1,
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'val_corr': val_metrics['corr'],
            'val_nrmse': val_metrics['nrmse'],
            'val_mape_score': val_metrics['mape_score'],
            'val_acc_like_0.1': val_metrics['acc_like_0.1'],
            'val_acc_like_0.2': val_metrics['acc_like_0.2'],
            'val_acc_like_0.3': val_metrics['acc_like_0.3'],
            'val_acc_like_0.4': val_metrics['acc_like_0.4'],
            'val_acc_like_0.5': val_metrics['acc_like_0.5'],
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(log_entry)
        
        # Best model 저장 (메모리에만)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Console 출력 (10 epoch마다)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{EPOCHS}] "
                  f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
                  f"R²: {val_metrics['r2']:.4f} | "
                  f"Corr: {val_metrics['corr']:.4f}")
    
    # ==================== Test Prediction (Best Model) ====================
    print(f"\n✓ Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    
    # Best model 로드
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Test 예측
    X_test_tensor = torch.tensor(X_test).to(DEVICE)
    with torch.no_grad():
        test_global_pred = model(X_test_tensor)
        test_pred_denorm = test_global_pred.cpu().numpy() * y_global_std + y_global_mean
    
    # OOF 저장
    oof_preds[test_idx] = test_pred_denorm
    
    print(f"✓ Fold {fold + 1} 완료! Test predictions saved to OOF.")

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

# OOF NPZ 저장
os.makedirs(os.path.dirname(OOF_PATH), exist_ok=True)
np.savez_compressed(OOF_PATH, pred=oof_preds)
print(f"✓ OOF 예측 저장: {OOF_PATH}")
print(f"  pred shape: {oof_preds.shape}")

# 전체 OOF 성능 계산
y_global_all = y_global
oof_metrics = calculate_metrics(y_global_all, oof_preds)

print(f"\n{'='*70}")
print(f"🏆 Overall OOF Performance (10-Fold Average)")
print(f"{'='*70}")
print(f"  MSE:             {oof_metrics['mse']:.2f}")
print(f"  MAE:             {oof_metrics['mae']:.2f}")
print(f"  RMSE:            {oof_metrics['rmse']:.2f}")
print(f"  R² Score:        {oof_metrics['r2']:.4f}")
print(f"  Correlation:     {oof_metrics['corr']:.4f}")
print(f"  NRMSE:           {oof_metrics['nrmse']:.4f}")
print(f"  MAPE Score:      {oof_metrics['mape_score']:.4f}")
print(f"  Acc-like (10%):  {oof_metrics['acc_like_0.1']:.4f}")
print(f"  Acc-like (20%):  {oof_metrics['acc_like_0.2']:.4f}")
print(f"  Acc-like (30%):  {oof_metrics['acc_like_0.3']:.4f}")
print(f"  Acc-like (40%):  {oof_metrics['acc_like_0.4']:.4f}")
print(f"  Acc-like (50%):  {oof_metrics['acc_like_0.5']:.4f}")
print(f"{'='*70}")

print("\n✅ 학습 완료!")
