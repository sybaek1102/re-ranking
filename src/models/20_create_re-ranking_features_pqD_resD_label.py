import numpy as np
import os

# =====================================================================
# 파일 경로 설정
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data")
INPUT_DIR = os.path.join(DATA_DIR, "input")

# 입력 파일
ORIGINAL_FEATURE_PATH = os.path.join(INPUT_DIR, "03_re-ranking_features_pqD_residual.npz")
NEW_LABEL_PATH = os.path.join(INPUT_DIR, "01_re-ranking_label.npz")

# 출력 파일
OUTPUT_FEATURE_PATH = os.path.join(INPUT_DIR, "03_re-ranking_features_pqD_residual_label.npz")

print("="*70)
print("📂 Label 교체 - Re-ranking Feature 생성")
print("="*70)

# =====================================================================
# 1. 데이터 로드
# =====================================================================
print("\n1️⃣  데이터 로드")

# 원본 feature 로드 (10000, 33) - 32 features + 1 label
with np.load(ORIGINAL_FEATURE_PATH) as f:
    original_data = f['data']  # (10000, 33)

print(f"✓ Original Data Shape: {original_data.shape}")

# Feature만 추출 (Label은 새로운 것을 사용할 예정)
X_original = original_data[:, :-1]  # (10000, 32)
y_original = original_data[:, -1:]  # (10000, 1) - 기존 label

print(f"✓ Original Features Shape: {X_original.shape}")
print(f"✓ Original Label Shape: {y_original.shape}")

# 새로운 Label 로드
with np.load(NEW_LABEL_PATH) as f:
    new_labels = f['data']  # (10000, 1)

print(f"✓ New Labels Shape: {new_labels.shape}")

# =====================================================================
# 2. Label 분포 비교
# =====================================================================
print("\n2️⃣  Label 분포 비교")

print(f"\n[기존 Label]")
print(f"  - Label 0: {np.sum(y_original == 0)}")
print(f"  - Label 1: {np.sum(y_original == 1)}")

print(f"\n[새로운 Label]")
print(f"  - Label 0: {np.sum(new_labels == 0)}")
print(f"  - Label 1: {np.sum(new_labels == 1)}")

# =====================================================================
# 3. 새로운 Label로 교체
# =====================================================================
print("\n3️⃣  새로운 Label로 교체")

# Features + New Label 결합
final_data = np.hstack([X_original, new_labels])  # (10000, 33)

print(f"✓ Final Data Shape: {final_data.shape}")
print(f"✓ Features: {final_data[:, :-1].shape}")
print(f"✓ Labels: {final_data[:, -1:].shape}")

# =====================================================================
# 4. 저장
# =====================================================================
print("\n4️⃣  저장")

# 저장
np.savez_compressed(OUTPUT_FEATURE_PATH, data=final_data)

print(f"\n✅ 파일 저장 완료: {OUTPUT_FEATURE_PATH}")

# =====================================================================
# 5. 검증
# =====================================================================
print("\n" + "="*70)
print("5️⃣  저장된 파일 검증")
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

# Feature 값이 동일한지 확인
features_match = np.allclose(X_original, loaded_data[:, :-1])
labels_match = np.allclose(new_labels, loaded_data[:, -1:])

print(f"\n✓ Verification:")
print(f"   - Features preserved: {'✅ YES' if features_match else '❌ NO'}")
print(f"   - Labels replaced: {'✅ YES' if labels_match else '❌ NO'}")

print("\n" + "="*70)
print("[파일 구성 (33 dims)]")
print("  - PQ Distance Features:     16 dims (||Q-P||²)")
print("  - Residual Features:        16 dims (||X-P||² - 2(Q-P)·(X-P))")
print("  - Label:                     1 dim (01_re-ranking_label.npz 사용)")
print("="*70)

print("\n✅ 모든 작업 완료!")
