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

FEATURE_PATH = os.path.join(INPUT_DIR, "12_residual_features_quantized.npz")
LABEL_PATH = os.path.join(INPUT_DIR, "12_residual_label_quantized.npz")

# =====================================================================
# 하이퍼파라미터
# =====================================================================
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_FOLDS = 10

# 모델 구조
SHARED_HIDDEN = 32
EMBED_DIM = 8
GLOBAL_HIDDEN = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")
print(f"📁 {NUM_FOLDS}-Fold OOF Training")

# =====================================================================
# Feature 조합 정의
# =====================================================================
FEATURE_CONFIGS = [
    {
        'name': 'int8_norm',
        'indices': list(range(11)),
        'log_path': os.path.join(OUTPUT_DIR, "logs", "12_residual_mlp_quantization_int8_norm_oof.csv"),
        'oof_path': os.path.join(OUTPUT_DIR, "oof", "12_residual_mlp_quantization_int8_norm_oof.npz"),
        'description': 'INT8 Norm Features'
    },
    {
        'name': 'int8_norm_scaled',
        'indices': [0,1,2,3,4,5,6,7,8,9,10,14,15,16],
        'log_path': os.path.join(OUTPUT_DIR, "logs", "12_residual_mlp_quantization_int8_norm_scaled_oof.csv"),
        'oof_path': os.path.join(OUTPUT_DIR, "oof", "12_residual_mlp_quantization_int8_norm_scaled_oof.npz"),
        'description': 'INT8 Norm Scaled Features'
    },
    {
        'name': 'int4_norm_scaled',
        'indices': [0,1,2,3,4,5,6,7,8,9,10,17,18,19],
        'log_path': os.path.join(OUTPUT_DIR, "logs", "12_residual_mlp_quantization_int4_norm_scaled_oof.csv"),
        'oof_path': os.path.join(OUTPUT_DIR, "oof", "12_residual_mlp_quantization_int4_norm_scaled_oof.npz"),
        'description': 'INT4 Norm Scaled Features'
    }
]

# =====================================================================
# 1. 데이터 로드
# =====================================================================
print("\n" + "="*70)
print("1️⃣  데이터 로드")
print("="*70)

if not os.path.exists(FEATURE_PATH) or not os.path.exists(LABEL_PATH):
    print(f"❌ Error: 파일을 찾을 수 없습니다.")
    print(f"   Feature: {FEATURE_PATH}")
    print(f"   Label: {LABEL_PATH}")
    sys.exit()

X_full_np = np.load(FEATURE_PATH)["data"].astype(np.float32)
y_np = np.load(LABEL_PATH)["data"].astype(np.float32)

print(f"✓ Full Feature Shape: {X_full_np.shape}")
print(f"✓ Label Shape: {y_np.shape}")

# Global labels
y_global = np.sum(y_np, axis=1)
print(f"✓ Global Label Shape: {y_global.shape}")

print(f"\n📊 Global Label Statistics:")
print(f"   Mean: {y_global.mean():.2f}")
print(f"   Std:  {y_global.std():.2f}")
print(f"   Min:  {y_global.min():.2f}")
print(f"   Max:  {y_global.max():.2f}")

# Target 정규화 통계
y_global_mean = y_global.mean()
y_global_std = y_global.std()

print(f"\n✓ Normalization Stats:")
print(f"   Global: mean={y_global_mean:.2f}, std={y_global_std:.2f}")

# =====================================================================
# 2. Fold Split 생성
# =====================================================================
print("\n" + "="*70)
print("2️⃣  10-Fold Split 생성")
print("="*70)

num_samples = len(X_full_np)
all_indices = np.arange(num_samples)
fold_chunks = np.array_split(all_indices, NUM_FOLDS)

print(f"✓ Total Samples: {num_samples}")
print(f"✓ Samples per Fold: ~{len(fold_chunks[0])}")

# =====================================================================
# 3. Model 정의
# =====================================================================
class SingleTaskDistancePredictor(nn.Module):
    def __init__(self, feature_dim):
        super(SingleTaskDistancePredictor, self).__init__()
        
        self.feature_dim = feature_dim
        self.input_norm = nn.BatchNorm1d(feature_dim)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, SHARED_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(SHARED_HIDDEN, EMBED_DIM),
            nn.LeakyReLU(0.1)
        )
        
        global_input_dim = 16 * EMBED_DIM
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, GLOBAL_HIDDEN),
            nn.LeakyReLU(0.1),
            nn.Linear(GLOBAL_HIDDEN, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.reshape(-1, self.feature_dim)
        x_norm = self.input_norm(x_flat)
        embeddings = self.shared_mlp(x_norm)
        global_input = embeddings.reshape(batch_size, -1)
        global_pred = self.global_mlp(global_input)
        return global_pred

# =====================================================================
# 4. Metric 계산 함수
# =====================================================================
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    y_range = y_true.max() - y_true.min()
    nrmse = 1 - (rmse / y_range) if y_range > 0 else 0
    
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    mape_score = 1 / (1 + mape / 100)
    
    y_std = y_true.std()
    acc_like_0_1 = np.mean(np.abs(y_true - y_pred) < y_std * 0.1)
    acc_like_0_2 = np.mean(np.abs(y_true - y_pred) < y_std * 0.2)
    acc_like_0_3 = np.mean(np.abs(y_true - y_pred) < y_std * 0.3)
    acc_like_0_4 = np.mean(np.abs(y_true - y_pred) < y_std * 0.4)
    acc_like_0_5 = np.mean(np.abs(y_true - y_pred) < y_std * 0.5)
    
    return {
        'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'corr': corr,
        'nrmse': nrmse, 'mape_score': mape_score,
        'acc_like_0.1': acc_like_0_1, 'acc_like_0.2': acc_like_0_2,
        'acc_like_0.3': acc_like_0_3, 'acc_like_0.4': acc_like_0_4,
        'acc_like_0.5': acc_like_0_5
    }

# =====================================================================
# 5. 각 Feature 조합별 10-Fold OOF 학습
# =====================================================================
for config in FEATURE_CONFIGS:
    print("\n" + "="*70)
    print(f"🚀 Training: {config['name']} (10-Fold OOF)")
    print(f"   {config['description']}")
    print("="*70)
    
    # Feature 선택
    feature_indices = config['indices']
    X_selected = X_full_np[:, :, feature_indices]
    feature_dim = len(feature_indices)
    
    print(f"✓ Selected Feature Shape: {X_selected.shape}")
    print(f"✓ Feature Dimension: {feature_dim}")
    
    # OOF 예측 저장
    oof_preds = np.zeros((num_samples, 1), dtype=np.float32)
    history = []
    
    # 10-Fold Loop
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*70}")
        print(f"📂 Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*70}")
        
        # Fold Split
        test_idx = fold_chunks[fold]
        val_idx = fold_chunks[(fold + 1) % NUM_FOLDS]
        train_chunks = []
        for i in range(NUM_FOLDS):
            if i != fold and i != (fold + 1) % NUM_FOLDS:
                train_chunks.append(fold_chunks[i])
        train_idx = np.concatenate(train_chunks)
        
        print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Data 추출
        X_train = X_selected[train_idx]
        X_val = X_selected[val_idx]
        X_test = X_selected[test_idx]
        
        y_global_train = (y_global[train_idx] - y_global_mean) / y_global_std
        y_global_val = (y_global[val_idx] - y_global_mean) / y_global_std
        y_global_val_original = y_global[val_idx]
        
        # DataLoader
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
        
        # Model 초기화
        model = SingleTaskDistancePredictor(feature_dim).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        # Epoch Loop
        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            train_loss_sum = 0
            
            for batch_X, batch_y_global in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y_global = batch_y_global.to(DEVICE)
                
                optimizer.zero_grad()
                global_pred = model(batch_X)
                loss = nn.MSELoss()(global_pred, batch_y_global)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)
            
            # Validation
            model.eval()
            val_loss_sum = 0
            all_val_preds = []
            
            with torch.no_grad():
                for batch_X, batch_y_global in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y_global = batch_y_global.to(DEVICE)
                    
                    global_pred = model(batch_X)
                    loss = nn.MSELoss()(global_pred, batch_y_global)
                    val_loss_sum += loss.item()
                    
                    global_pred_denorm = global_pred.cpu().numpy() * y_global_std + y_global_mean
                    all_val_preds.append(global_pred_denorm)
            
            all_val_preds = np.concatenate(all_val_preds)
            avg_val_loss = val_loss_sum / len(val_loader)
            
            # Metrics
            val_metrics = calculate_metrics(y_global_val_original, all_val_preds)
            scheduler.step(avg_val_loss)
            
            # Log
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
            
            # Best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Console (10 epoch마다)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch [{epoch:3d}/{EPOCHS}] "
                      f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
                      f"R²: {val_metrics['r2']:.4f} | "
                      f"Corr: {val_metrics['corr']:.4f}")
        
        # Test Prediction (Best Model)
        print(f"\n✓ Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
        
        model.load_state_dict(best_model_state)
        model.eval()
        
        X_test_tensor = torch.tensor(X_test).to(DEVICE)
        with torch.no_grad():
            test_global_pred = model(X_test_tensor)
            test_pred_denorm = test_global_pred.cpu().numpy() * y_global_std + y_global_mean
        
        oof_preds[test_idx] = test_pred_denorm
        print(f"✓ Fold {fold + 1} 완료! Test predictions saved to OOF.")
    
    # 결과 저장
    os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    df_history = pd.DataFrame(history)
    df_history.to_csv(config['log_path'], index=False)
    print(f"\n✓ 학습 로그 저장: {config['log_path']}")
    
    os.makedirs(os.path.dirname(config['oof_path']), exist_ok=True)
    np.savez_compressed(config['oof_path'], pred=oof_preds)
    print(f"✓ OOF 예측 저장: {config['oof_path']}")
    print(f"  pred shape: {oof_preds.shape}")
    
    # Overall OOF 성능
    oof_metrics = calculate_metrics(y_global, oof_preds)
    
    print(f"\n{'='*70}")
    print(f"🏆 Overall OOF Performance ({config['name']})")
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

print("\n" + "="*70)
print("✅ 모든 Feature 조합 10-Fold OOF 학습 완료!")
print("="*70)
