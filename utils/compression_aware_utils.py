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
    
    # 추가
    df.columns = df.columns.str.strip()

    if "QP" not in df.columns and "qp" in df.columns:
        df["QP"] = df["qp"]


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

    # 평균값
    avg_bit_trust = df["bit_trust"].mean()

    if debug:
        print("\n[DEBUG] === Bit-based Trust ===")
        print(f"📊 Bits range: {min_bits:.1f} → {max_bits:.1f}")
        print(f"Max scale: {max_value}")
        print(f"📈 Average bit_trust: {avg_bit_trust:.4f}")  # debug일 때만 출력
        print(df[["Frame_ID", bit_col, "bit_trust"]].head(10).to_string(index=False))

    bit_trust_dict = df.set_index("Frame_ID")["bit_trust"].to_dict()
    
    # dict와 평균값 모두 반환
    return bit_trust_dict, avg_bit_trust

    # return df.set_index("Frame_ID")["bit_trust"].to_dict()


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
    
    # 추가
    df_qp.columns = df_qp.columns.str.strip()

    if "QP" not in df_qp.columns and "qp" in df_qp.columns:
        df_qp["QP"] = df_qp["qp"]

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

    
    avg_importance = df_train["importance"].mean()

    if debug:
        print("\n[DEBUG] === Frame Trust Metrics (Linear Scaled) ===")
        print(f"📊 QP range: {min_qp:.2f} → {max_qp:.2f}")
        print(f"📈 Average importance: {avg_importance:.4f}")
        print(df_train[["Frame_ID", qp_col, "importance"]].head(10).to_string(index=False))

    importance_dict = df_train.set_index("Train_ID")["importance"].to_dict()
    
    # dict와 평균값 모두 반환
    return importance_dict, avg_importance
    # return df_train.set_index("Train_ID")["importance"].to_dict()




# ================= Pose Gradient Scaling (Inlier-based) ================= #
def apply_inlier_weighted_pose_grad(viewpoint_cam, phase="local", p_mu=0.1):
    """
    Scales pose parameter gradients based on inlier ratio.
    phase: "local" or "global"
    p_mu: base hyperparameter controlling weight strength (e.g., 0.1~0.3)
    """

    # inlier ratio 가져오기 (없으면 1.0)
    inlier_ratio = getattr(viewpoint_cam, "inlier_ratio", 1.0)

    # phase에 따라 p_mu 조정 (local일 때 +0.2)
    p_mu_eff = p_mu + 0.2 if phase == "local" else p_mu

    # weight 계산 (불안정할수록 pose 영향 감소)
    w_pose = 1.0 - p_mu_eff * (1.0 - inlier_ratio)
    w_pose = max(0.0, min(1.0, w_pose))  # 안전한 범위 클램프

    # gradient scaling (pose 관련 파라미터에만 적용)
    for p in [getattr(viewpoint_cam, "cam_trans_delta", None),
              getattr(viewpoint_cam, "cam_rot_delta", None)]:
        if p is not None and p.grad is not None:
            p.grad.mul_(w_pose)

    # 로그 (옵션)
    print(f"[{phase.upper()} PoseGrad] uid={getattr(viewpoint_cam, 'uid', -1)} "
          f"inlier={inlier_ratio:.3f}, p_mu={p_mu_eff:.2f}, w_pose={w_pose:.3f}")
# ======================================================================== #
