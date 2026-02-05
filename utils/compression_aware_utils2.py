import pandas as pd
import numpy as np
import os
EPS = 1e-8

def sigmoid_temperature(x, T=0.15):
    """
    x: numpy array or pandas series in [0, 1]
    T: shared temperature (single hyperparameter)
    returns: sigmoid((x-0.5)/T) in (0,1)
    """
    z = (x - 0.5) / (T + EPS)
    # numerical stability clamp (optional but safe)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


# 1) Bit trust (sigmoid version)
def compute_bit_based_trust(qp_csv, max_value=1.0, T=0.15, debug=False):
    """
    [Compression-Aware] Bit-based frame trust mapping (Sigmoid)
    - bits 값을 min-max로 [0,1] 만든 뒤
    - sigmoid((x-0.5)/T) 적용
    - 마지막에 max_value로 스케일

    Returns:
        dict: {Frame_ID: bit_trust_score}
        float: average bit_trust
    """
    if not os.path.exists(qp_csv):
        raise FileNotFoundError(f"❌ QP CSV not found: {qp_csv}")

    df = pd.read_csv(qp_csv)

    if "Global_Frame_ID" in df.columns:
        df = df.rename(columns={"Global_Frame_ID": "Frame_ID"})

    bit_col_candidates = ["Bits", "bits", "FrameBits", "Frame_Bits"]
    bit_col = next((c for c in bit_col_candidates if c in df.columns), None)
    if bit_col is None:
        raise KeyError(f"❌ No bits column found in {qp_csv}")

    min_bits = df[bit_col].min()
    max_bits = df[bit_col].max()

    # min-max normalization -> [0,1]
    bit_norm = (df[bit_col] - min_bits) / (max_bits - min_bits + EPS)
    bit_norm = bit_norm.clip(0.0, 1.0)

    # sigmoid nonlinearity (shared T)
    df["bit_trust"] = sigmoid_temperature(bit_norm.to_numpy(), T=T)

    # apply max scale
    df["bit_trust"] = df["bit_trust"] * max_value

    avg_bit_trust = float(df["bit_trust"].mean())

    if debug:
        print("\n[DEBUG] === Bit-based Trust (Sigmoid) ===")
        print(f"📊 Bits range: {min_bits:.1f} → {max_bits:.1f}")
        print(f"T (shared): {T}")
        print(f"Max scale: {max_value}")
        print(f"📈 Average bit_trust: {avg_bit_trust:.4f}")
        print(df[["Frame_ID", bit_col, "bit_trust"]].head(10).to_string(index=False))

    bit_trust_dict = df.set_index("Frame_ID")["bit_trust"].to_dict()
    return bit_trust_dict, avg_bit_trust


# 2) QP trust (sigmoid version)
def load_frame_trust_metrics(qp_csv, T=0.15, debug=False):
    """
    [Compression-Aware] Frame trust metrics (QP only, Sigmoid)
    - QP 낮을수록 신뢰도 높음
    - (max_qp - QP) min-max 정규화로 [0,1]
    - sigmoid((x-0.5)/T) 적용

    Returns:
        dict: {Train_ID: importance_score}
        float: average importance
    """
    if not os.path.exists(qp_csv):
        raise FileNotFoundError(f"❌ QP CSV not found: {qp_csv}")

    df_qp = pd.read_csv(qp_csv)

    if "Global_Frame_ID" in df_qp.columns:
        df_qp = df_qp.rename(columns={"Global_Frame_ID": "Frame_ID"})

    if "Is_Test" in df_qp.columns:
        df_train = df_qp[df_qp["Is_Test"] == False].copy()
    else:
        df_train = df_qp.copy()

    df_train = df_train.sort_values("Frame_ID").reset_index(drop=True)
    df_train["Train_ID"] = df_train.index

    qp_col = "QP"
    if qp_col not in df_train.columns:
        raise KeyError("❌ Missing required QP column in QP CSV.")

    min_qp = df_train[qp_col].min()
    max_qp = df_train[qp_col].max()

    # linear norm -> [0,1]
    qp_norm = (max_qp - df_train[qp_col]) / (max_qp - min_qp + EPS)
    qp_norm = qp_norm.clip(0.0, 1.0)

    # sigmoid nonlinearity (shared T)
    df_train["importance"] = sigmoid_temperature(qp_norm.to_numpy(), T=T)

    avg_importance = float(df_train["importance"].mean())

    if debug:
        print("\n[DEBUG] === Frame Trust Metrics (Sigmoid) ===")
        print(f"📊 QP range: {min_qp:.2f} → {max_qp:.2f}")
        print(f"T (shared): {T}")
        print(f"📈 Average importance: {avg_importance:.4f}")
        print(df_train[["Frame_ID", qp_col, "importance"]].head(10).to_string(index=False))

    importance_dict = df_train.set_index("Train_ID")["importance"].to_dict()
    return importance_dict, avg_importance




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
