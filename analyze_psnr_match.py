import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ CSV 파일 로드
# ===============================
key_csv = "/workdir/outputs/free/grass_qp37_imp1/keypoint_match_log.csv"
psnr_csv = "/workdir/comp_log/grass_qp37_trustmap.csv"

df_key = pd.read_csv(key_csv)
df_psnr = pd.read_csv(psnr_csv)

# ===============================
# 2️⃣ Inlier 비율 계산
# ===============================
df_key["Inlier_Ratio"] = df_key["Num_Inliers"] / df_key["Num_Keypoints"]

# ===============================
# 3️⃣ PSNR 차이 계산
# ===============================
psnr_dict = dict(zip(df_psnr["Global_Frame_ID"], df_psnr["Y_PSNR"]))

df_key["Delta_Y_PSNR"] = df_key.apply(
    lambda row: psnr_dict.get(row["Frame_ID"], None) - psnr_dict.get(row["Prev_Frame_ID"], None)
    if (row["Frame_ID"] in psnr_dict and row["Prev_Frame_ID"] in psnr_dict)
    else None,
    axis=1
)

# 절대값 버전 추가
df_key["Abs_Delta_Y_PSNR"] = df_key["Delta_Y_PSNR"].abs()

# ===============================
# 4️⃣ 그래프 그리는 함수 (GOP 선 포함)
# ===============================
def plot_and_save(df, delta_col, title, output_path):
    fig, ax1 = plt.subplots(figsize=(20, 4))
    ax2 = ax1.twinx()

    # 왼쪽 Y축: Inlier Ratio
    ax1.plot(df["Frame_ID"], df["Inlier_Ratio"], "o-", color="tab:blue", label="Inlier Ratio")
    ax1.set_ylabel("Inlier Ratio", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # 오른쪽 Y축: ΔY_PSNR or |ΔY_PSNR|
    ax2.plot(df["Frame_ID"], df[delta_col], "s--", color="tab:red", label=title)
    ax2.set_ylabel("ΔY_PSNR (dB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # GOP 표시: 32 프레임마다 회색 수직선
    max_frame = int(df["Frame_ID"].max())
    for gop_start in range(0, max_frame + 1, 32):
        ax1.axvline(x=gop_start, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax1.text(
            gop_start + 1, ax1.get_ylim()[1] * 0.95,
            f"GOP {gop_start//32}",
            color="gray", fontsize=8, rotation=0, va="top"
        )

    # 제목 및 X축
    ax1.set_xlabel("Frame ID")
    ax1.set_title(title)

    # 범례
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 그래프 저장 완료: {output_path}")

# ===============================
# 5️⃣ 두 그래프 저장
# ===============================
plot_and_save(df_key, "Delta_Y_PSNR", "Inlier Ratio vs ΔY_PSNR (per Frame)", "inlier_vs_psnr_delta_imp1.png")
plot_and_save(df_key, "Abs_Delta_Y_PSNR", "Inlier Ratio vs |ΔY_PSNR| (per Frame)", "inlier_vs_psnr_abs_delta_imp1.png")
