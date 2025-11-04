import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd





import pandas as pd
import numpy as np
import os

# bit 

def compute_bit_based_trust(qp_csv, max_value=1.0, debug=False):
    """
    [Compression-Aware] Bit-based frame trust mapping
    - 프레임별 bits 값을 선형 스케일링하여 trust score 계산
    - 최소값 → 0.0, 최대값 → max_value 로 매핑

    Args:
        qp_csv (str): QP 로그 CSV 경로 (필수)
        max_value (float): 최대 trust 스케일 (기본값 0.5)
        debug (bool): 디버그 출력 여부

    Returns:
        dict: {Frame_ID: bit_trust_score}
    """
    if not os.path.exists(qp_csv):
        raise FileNotFoundError(f"❌ QP CSV not found: {qp_csv}")

    df = pd.read_csv(qp_csv)

    # Frame ID 표준화
    if "Global_Frame_ID" in df.columns:
        df = df.rename(columns={"Global_Frame_ID": "Frame_ID"})

    # bits 컬럼 확인
    bit_col_candidates = ["Bits", "bits", "FrameBits", "Frame_Bits"]
    bit_col = next((c for c in bit_col_candidates if c in df.columns), None)
    if bit_col is None:
        raise KeyError(f"❌ No bits column found in {qp_csv}")

    # min-max normalization
    min_bits = df[bit_col].min()
    max_bits = df[bit_col].max()
    # print('min_bits:', min_bits, 'max_bits:', max_bits)
    df["bit_trust"] = (df[bit_col] - min_bits) / (max_bits - min_bits + 1e-8)
    df["bit_trust"] = df["bit_trust"] * max_value  # 상한 스케일 적용


    if debug:
        print("\n[DEBUG] === Bit-based Trust ===")
        print(f"📊 Bits range: {min_bits:.1f} → {max_bits:.1f}")
        print(f"Max scale: {max_value}")
        print(df[["Frame_ID", bit_col, "bit_trust"]].head(10).to_string(index=False))

    return df.set_index("Frame_ID")["bit_trust"].to_dict()


# 2. QP only
def load_frame_trust_metrics(qp_csv, debug=False):
    """
    [Compression-Aware] Frame trust metrics (QP only, scaled 0~1)
    - QP 낮을수록 신뢰도 높음
    - QP 범위를 이용한 선형 정규화 (min_qp → 1, max_qp → 0)
    """

    if not os.path.exists(qp_csv):
        raise FileNotFoundError(f"❌ QP CSV not found: {qp_csv}")

    df_qp = pd.read_csv(qp_csv)

    if "Global_Frame_ID" in df_qp.columns:
        df_qp = df_qp.rename(columns={"Global_Frame_ID": "Frame_ID"})

    # train-only 필터
    if "Is_Test" in df_qp.columns:
        df_train = df_qp[df_qp["Is_Test"] == False].copy()
    else:
        df_train = df_qp.copy()

    df_train = df_train.sort_values("Frame_ID").reset_index(drop=True)
    df_train["Train_ID"] = df_train.index

    # ✅ QP 컬럼 탐색
    qp_col_candidates = ["QP"]
    qp_col = next((c for c in qp_col_candidates if c in df_train.columns), None)
    if qp_col is None:
        raise KeyError("❌ Missing required QP column in QP CSV.")

    # ✅ QP 기반 0~1 스케일링
    min_qp = df_train[qp_col].min()
    max_qp = df_train[qp_col].max()
    df_train["importance"] = (max_qp - df_train[qp_col]) / (max_qp - min_qp + 1e-8)

    # ✅ 안전하게 [0, 1] 범위 보정
    df_train["importance"] = df_train["importance"].clip(0.0, 1.0)

    if debug:
        print("\n[DEBUG] === Frame Trust Metrics (Linear Scaled) ===")
        print(f"📊 QP range: {min_qp:.2f} → {max_qp:.2f}")
        print(df_train[["Frame_ID", qp_col, "importance"]].head(10).to_string(index=False))

    return df_train.set_index("Train_ID")["importance"].to_dict()

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 분석할 scene 목록
scene_list = ["grass", "hydrant", "lab", "pillar", "road", "sky", "stair"]
base_dir = "./comp_log"
csv_suffix = "_qp32_trustmap.csv"

# 결과 저장용
summary_list = []

for scene in scene_list:
    csv_path = os.path.join(base_dir, f"{scene}{csv_suffix}")
    if not os.path.exists(csv_path):
        print(f"⚠️ Skip: {csv_path} not found.")
        continue

    print(f"\n==============================")
    print(f"📁 Scene: {scene}")
    print("==============================")

    # 1️⃣ 함수 실행
    bit_trust_dict = compute_bit_based_trust(csv_path, max_value=0.5, debug=False)
    qp_trust_dict = load_frame_trust_metrics(csv_path, debug=False)

    # 2️⃣ DataFrame 변환 및 정리
    df_trust = pd.DataFrame({
        "bit_trust": pd.Series(bit_trust_dict),
        "qp_importance": pd.Series(qp_trust_dict)
    }).dropna()

    if len(df_trust) == 0:
        print(f"⚠️ No overlapping Frame_IDs for {scene}, skipping.")
        continue

    # 3️⃣ 통계 요약
    desc_bit = df_trust["bit_trust"].describe()
    desc_qp = df_trust["qp_importance"].describe()
    corr = df_trust["bit_trust"].corr(df_trust["qp_importance"])

    print("=== Bit-based Trust Summary ===")
    print(desc_bit, "\n")
    print("=== QP-based Importance Summary ===")
    print(desc_qp, "\n")
    print(f"🔹 Correlation: {corr:.3f}")

    # 결과 기록
    summary_list.append({
        "Scene": scene,
        "BitTrust_Mean": desc_bit["mean"],
        "BitTrust_Std": desc_bit["std"],
        "QPTrust_Mean": desc_qp["mean"],
        "QPTrust_Std": desc_qp["std"],
        "Correlation": corr,
        "Samples": len(df_trust)
    })

    # 4️⃣ 시각화
    plt.figure(figsize=(10,4))
    sns.kdeplot(df_trust["bit_trust"], label="Bit Trust", fill=True, alpha=0.4)
    sns.kdeplot(df_trust["qp_importance"], label="QP Importance", fill=True, alpha=0.4)
    plt.title(f"[{scene.upper()}] Trust Metric Distribution")
    plt.xlabel("Trust Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 5️⃣ 전체 요약 표로 보기
df_summary = pd.DataFrame(summary_list)
print("\n========== 📊 Overall Summary ==========")
print(df_summary.round(4).to_string(index=False))

# 6️⃣ 전체 scene 비교 시각화
if not df_summary.empty:
    plt.figure(figsize=(8,5))
    sns.barplot(data=df_summary, x="Scene", y="Correlation", palette="viridis")
    plt.title("Correlation between Bit Trust and QP Importance per Scene")
    plt.xlabel("Scene")
    plt.ylabel("Pearson Correlation")
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()
