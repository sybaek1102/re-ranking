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
ORIGINAL_FEATURE_PATH = os.path.join(INPUT_DIR, "03_re-ranking_features_pqD_residual.npz") # pqD
OOF_PRED_PATH = os.path.join(OUTPUT_DIR, "oof", "14_residual_mlp_full_oof.npz") # resD
NEW_LABEL_PATH = os.path.join(INPUT_DIR, "01_re-ranking_label.npz")  # label - state -1 == label 1

# 출력 파일
OUTPUT_FEATURE_PATH = os.path.join(INPUT_DIR, "24_re-ranking_pqD_pred_resD_label.npz")

print("="*70)
print("📂 OOF 예측 기반 Re-ranking Feature 생성")
print("="*70)

# =====================================================================
# 1. 데이터 로드
# =====================================================================
print("\n1️⃣  데이터 로드")

# 원본 feature 로드 (N, 33) - 32 features + 1 label
with np.load(ORIGINAL_FEATURE_PATH) as f:
    original_data = f['data']  # (160000, 33)

print(f"✓ Original Data Shape: {original_data.shape}")

# Feature만 추출 (Label은 새로운 것을 사용할 예정)
X_original = original_data[:, :-1]  # (160000, 32)

print(f"✓ Original Features Shape: {X_original.shape}")

# 앞 16개 feature만 추출 (PQ Distance)
pq_dist_features = X_original[:, :16]  # (160000, 16)
print(f"✓ PQ Distance Features Shape: {pq_dist_features.shape}")

# 뒤 16개 feature (Residual 관련)
residual_features = X_original[:, 16:]  # (160000, 16)
print(f"✓ Residual Features Shape: {residual_features.shape}")

# OOF 예측 결과 로드
with np.load(OOF_PRED_PATH) as f:
    oof_preds = f['pred']  # (160000, 1)

print(f"✓ OOF Predictions Shape: {oof_preds.shape}")

# 새로운 Label 로드
with np.load(NEW_LABEL_PATH) as f:
    new_labels = f['data']  # (10000, 1)

print(f"✓ New Labels Shape: {new_labels.shape}")

# =====================================================================
# 2. 데이터 재구성 확인
# =====================================================================
print("\n2️⃣  데이터 형태 확인 및 재구성")

# 원본 데이터가 이미 (10000, 32) 형태로 reshape 필요
pq_dist_features = pq_dist_features.reshape(10000, 16)
residual_features = residual_features.reshape(10000, 16)

print(f"✓ PQ Distance Features Reshaped: {pq_dist_features.shape}")
print(f"✓ Residual Features Reshaped: {residual_features.shape}")

# OOF 예측을 (10000, 16) 형태로 reshape
oof_preds_reshaped = oof_preds.reshape(10000, 16)  # (10000, 16)

print(f"✓ OOF Preds Reshaped: {oof_preds_reshaped.shape}")

# =====================================================================
# 3. 새로운 Feature 계산
# =====================================================================
print("\n3️⃣  새로운 Feature 계산")

# ||R||^2 - 2 * predicted(dot(Q-C, R))
# residual_features는 (10000, 16) - 이미 ||R||^2 값
# oof_preds_reshaped는 (10000, 16) - predicted(dot(Q-C, R))

new_residual_features = residual_features - 2 * oof_preds_reshaped  # (10000, 16)

print(f"✓ New Residual Features Shape: {new_residual_features.shape}")
print(f"✓ New Residual Features - Mean: {new_residual_features.mean():.4f}, Std: {new_residual_features.std():.4f}")

# =====================================================================
# 4. Feature 병합
# =====================================================================
print("\n4️⃣  Feature 병합")

# PQ Distance (16) + New Residual (16) = 32 features
final_features = np.hstack([pq_dist_features, new_residual_features])  # (10000, 32)

print(f"✓ Final Features Shape: {final_features.shape}")

# Label은 새로 로드한 것 사용
final_labels = new_labels  # (10000, 1)

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
print("  - PQ Distance Features:     16 dims (앞 16개)")
print("  - OOF-based Residual Dist:  16 dims (||R||² - 2*pred(⟨Q-C,R⟩))")
print("  - Label:                     1 dim (01_re-ranking_label.npz 사용)")
print("="*70)

print("\n✅ 모든 작업 완료!")