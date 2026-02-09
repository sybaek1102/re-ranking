import numpy as np
import os

# =====================================================================
# 파일 경로 설정
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# 입력 파일
ORIGINAL_FEATURE_PATH = os.path.join(INPUT_DIR, "03_re-ranking_features_pqD_residual.npz")
OOF_PRED_PATH = os.path.join(OUTPUT_DIR, "oof", "12_residual_mlp_quantization_int8_norm_scaled_oof.npz")

# 출력 파일
OUTPUT_FEATURE_PATH = os.path.join(INPUT_DIR, "22_re-ranking_pqD_pred_resD_int8_norm_scaled.npz")

print("="*70)
print("📂 OOF 예측 기반 Re-ranking Feature 생성")
print("="*70)

# =====================================================================
# 1. 데이터 로드
# =====================================================================
print("\n1️⃣  데이터 로드")

# 원본 feature 로드
with np.load(ORIGINAL_FEATURE_PATH) as f:
    original_data = f['data']

print(f"✓ Original Data Shape: {original_data.shape}")

# Feature와 Label 분리
X_original = original_data[:, :-1]
y_original = original_data[:, -1:]

print(f"✓ Original Features Shape: {X_original.shape}")
print(f"✓ Original Label Shape: {y_original.shape}")

# 앞 16개 feature만 추출 (PQ Distance)
pq_dist_features = X_original[:, :16]
print(f"✓ PQ Distance Features Shape: {pq_dist_features.shape}")

# 뒤 16개 feature (Residual 관련 - 사용하지 않음)
residual_features = X_original[:, 16:]
print(f"✓ Original Residual Features Shape: {residual_features.shape} (will be replaced)")

# OOF 예측 결과 로드
with np.load(OOF_PRED_PATH) as f:
    oof_preds = f['pred']

print(f"✓ OOF Predictions Shape: {oof_preds.shape}")

# =====================================================================
# 2. 데이터 재구성
# =====================================================================
print("\n2️⃣  데이터 형태 확인 및 재구성")

num_samples = original_data.shape[0]
print(f"✓ Number of samples in original data: {num_samples}")

# OOF 예측을 (10000, 16) 형태로 reshape
oof_preds_reshaped = oof_preds.reshape(10000, 16)
print(f"✓ OOF Preds Reshaped: {oof_preds_reshaped.shape}")

# 원본 데이터 형태에 따라 처리
if num_samples == 10000:
    print(f"✓ Original data는 이미 (10000, 33) 형태입니다.")
    pq_dist_reshaped = pq_dist_features  # (10000, 16)
    final_labels = y_original  # (10000, 1)
    
elif num_samples == 160000:
    print(f"✓ Original data는 (160000, 33) = (10000 queries × 16 candidates) 형태입니다.")
    # PQ Distance reshape
    pq_dist_reshaped = pq_dist_features.reshape(10000, 16)
    # Label reshape - 각 query당 첫 번째 label만
    final_labels = y_original.reshape(10000, 16)[:, 0:1]
    
else:
    raise ValueError(f"Unexpected number of samples: {num_samples}")

print(f"✓ PQ Distance Reshaped: {pq_dist_reshaped.shape}")
print(f"✓ Final Labels Shape: {final_labels.shape}")

# =====================================================================
# 3. 새로운 Feature 설정
# =====================================================================
print("\n3️⃣  새로운 Residual Feature 설정")

# OOF 예측값을 그대로 새로운 residual feature로 사용
new_residual_features = oof_preds_reshaped  # (10000, 16)

print(f"✓ New Residual Features Shape: {new_residual_features.shape}")
print(f"✓ New Residual Features - Mean: {new_residual_features.mean():.4f}, Std: {new_residual_features.std():.4f}")
print(f"✓ New Residual Features - Min: {new_residual_features.min():.4f}, Max: {new_residual_features.max():.4f}")

# =====================================================================
# 4. Feature 병합
# =====================================================================
print("\n4️⃣  Feature 병합")

# PQ Distance (16) + OOF Predicted Residual (16) = 32 features
final_features = np.hstack([pq_dist_reshaped, new_residual_features])  # (10000, 32)

print(f"✓ Final Features Shape: {final_features.shape}")
print(f"✓ Final Labels Shape: {final_labels.shape}")

# Label 분포 확인
print(f"✓ Label Distribution - 0: {np.sum(final_labels == 0)}, 1: {np.sum(final_labels == 1)}")

# =====================================================================
# 5. 최종 데이터 결합 및 저장
# =====================================================================
print("\n5️⃣  최종 데이터 결합 및 저장")

# Features + Label 결합
final_data = np.hstack([final_features, final_labels])  # (10000, 33)

print(f"✓ Final Data Shape: {final_data.shape}")

# 저장
os.makedirs(os.path.dirname(OUTPUT_FEATURE_PATH), exist_ok=True)
np.savez_compressed(OUTPUT_FEATURE_PATH, data=final_data)

print(f"\n✅ 파일 저장 완료: {OUTPUT_FEATURE_PATH}")

# =====================================================================
# 6. 검증
# =====================================================================
print("\n" + "="*70)
print("6️⃣  저장된 파일 검증")
print("="*70)

with np.load(OUTPUT_FEATURE_PATH) as f:
    loaded_data = f['data']

print(f"\n✓ Loaded Data Shape: {loaded_data.shape}")
print(f"✓ Expected Shape: (10000, 33)")
print(f"✓ Match: {'✅ OK' if loaded_data.shape == (10000, 33) else '❌ MISMATCH'}")

print(f"\n✓ Feature Statistics:")
print(f"   - Features Shape: {loaded_data[:, :-1].shape}")
print(f"   - Features Mean: {loaded_data[:, :-1].mean():.4f}")
print(f"   - Features Std: {loaded_data[:, :-1].std():.4f}")

print(f"\n✓ Label Statistics:")
print(f"   - Label Shape: {loaded_data[:, -1:].shape}")
print(f"   - Label 0: {np.sum(loaded_data[:, -1] == 0)}")
print(f"   - Label 1: {np.sum(loaded_data[:, -1] == 1)}")

print("\n" + "="*70)
print("[Feature 구성 (33 dims)]")
print("  - PQ Distance Features:         16 dims (앞 16개)")
print("  - OOF Predicted Residual Dist:  16 dims (MLP 예측값)")
print("  - Label:                         1 dim")
print("="*70)

print("\n✅ 모든 작업 완료!")