import pandas as pd
import numpy as np
import os

# ============================================
# Load and prepare
# ============================================
df = pd.read_csv("/workdir/outputs/free/grass_debug_2/gaussian_stats_log.csv")
os.makedirs("/workdir/analyze_codes/plots_notcomp_gs", exist_ok=True)

# ============================================
# 1️⃣ 전체 통계 요약
# ============================================
summary = {
    "Total Iterations": df["Iteration"].max(),
    "Total Frames": df["Frame_ID"].nunique(),
    "Final Num_Gaussians": df["Num_Gaussians"].iloc[-1],
    "Mean Num_New (per iteration)": df["Num_New"].mean(),
    "Mean Scale": df["Mean_Scale"].mean(),
    "Std(Scale)": df["Mean_Scale"].std(),
    "Mean Opacity": df["Mean_Opacity"].mean(),
    "Std(Opacity)": df["Mean_Opacity"].std(),
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv("/workdir/analyze_codes/plots_notcomp_gs/summary_stats.csv", index=False)
print("📊 Summary Statistics:\n", summary_df, "\n")

# ============================================
# 2️⃣ Gaussian 성장 분석
# ============================================
df["Gauss_Growth_Rate"] = df["Num_Gaussians"].diff()
growth_mean = df["Gauss_Growth_Rate"].mean()
growth_std = df["Gauss_Growth_Rate"].std()
growth_plateau_iter = df.loc[df["Gauss_Growth_Rate"].abs() < 0.01 * df["Gauss_Growth_Rate"].max(), "Iteration"].min()

print(f"📈 평균 Gaussian 증가량: {growth_mean:.2f} ± {growth_std:.2f}")
print(f"⚖️ 안정화 시작 Iteration: {growth_plateau_iter}\n")

# ============================================
# 3️⃣ Scale 수렴도 분석
# ============================================
scale_start = df["Mean_Scale"].iloc[:10].mean()
scale_end = df["Mean_Scale"].iloc[-10:].mean()
scale_reduction = (scale_start - scale_end) / scale_start * 100

scale_stability = df["Std_Scale"].iloc[-50:].mean()
print(f"🔍 Scale 감소율: {scale_reduction:.2f}%")
print(f"📏 마지막 50 iter 평균 Std(Scale): {scale_stability:.5f}\n")

# ============================================
# 4️⃣ Opacity 안정도 분석
# ============================================
opacity_start = df["Mean_Opacity"].iloc[:10].mean()
opacity_end = df["Mean_Opacity"].iloc[-10:].mean()
opacity_gain = (opacity_end - opacity_start) / opacity_start * 100

opacity_stability = df["Std_Opacity"].iloc[-50:].mean()
print(f"🌫️ Opacity 증가율: {opacity_gain:.2f}%")
print(f"📏 마지막 50 iter 평균 Std(Opacity): {opacity_stability:.5f}\n")

# ============================================
# 5️⃣ Scale–Opacity 상관관계
# ============================================
corr = df["Mean_Scale"].corr(df["Mean_Opacity"])
print(f"🔗 Scale–Opacity 상관계수: {corr:.3f}")

# ============================================
# 6️⃣ Frame별 Gaussian 생성 평균
# ============================================
frame_summary = df.groupby("Frame_ID")[["Num_New", "Mean_Scale", "Mean_Opacity"]].mean().reset_index()
frame_summary.to_csv("/workdir/analyze_codes/plots_notcomp_gs/framewise_stats.csv", index=False)
print("🧩 Frame별 평균 통계가 '/workdir/analyze_codes/plots_notcomp_gs/framewise_stats.csv'로 저장되었습니다.\n")

# ============================================
# 저장된 파일 안내
# ============================================
print("✅ 분석 완료:")
print(" - /workdir/analyze_codes/plots_notcomp_gs/summary_stats.csv : 전체 요약 통계")
print(" - /workdir/analyze_codes/plots_notcomp_gs/framewise_stats.csv : 프레임별 세부 통계")

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # ============================================
# # Load and prepare
# # ============================================
# df = pd.read_csv("/workdir/outputs/free/grass_debug_2/gaussian_stats_log.csv")

# # 저장 폴더 생성
# os.makedirs("/workdir/analyze_codes/plots_notcomp_gs", exist_ok=True)

# # ============================================
# # 1️⃣ Gaussian 개수 증가 추이
# # ============================================
# plt.figure(figsize=(7,4))
# plt.plot(df["Iteration"], df["Num_Gaussians"], label="Total Gaussians", color='tab:blue')
# plt.xlabel("Iteration")
# plt.ylabel("Num_Gaussians")
# plt.title("Gaussian Growth Over Iterations (Uncompressed)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("/workdir/analyze_codes/plots_notcomp_gs/gaussian_growth.png", dpi=300)
# plt.close()

# # ============================================
# # 2️⃣ Gaussian Scale 변화
# # ============================================
# plt.figure(figsize=(7,4))
# plt.plot(df["Iteration"], df["Mean_Scale"], color='tab:green', label="Mean Scale")
# plt.fill_between(df["Iteration"],
#                  df["Mean_Scale"] - df["Std_Scale"],
#                  df["Mean_Scale"] + df["Std_Scale"],
#                  alpha=0.2, color='tab:green', label="±1 Std")
# plt.xlabel("Iteration")
# plt.ylabel("Mean_Scale")
# plt.title("Gaussian Scale Evolution (Uncompressed)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/workdir/analyze_codes/plots_notcomp_gs/scale_evolution.png", dpi=300)
# plt.close()

# # ============================================
# # 3️⃣ Gaussian Opacity 변화
# # ============================================
# plt.figure(figsize=(7,4))
# plt.plot(df["Iteration"], df["Mean_Opacity"], color='tab:orange', label="Mean Opacity")
# plt.fill_between(df["Iteration"],
#                  df["Mean_Opacity"] - df["Std_Opacity"],
#                  df["Mean_Opacity"] + df["Std_Opacity"],
#                  alpha=0.2, color='tab:orange', label="±1 Std")
# plt.xlabel("Iteration")
# plt.ylabel("Mean_Opacity")
# plt.title("Gaussian Opacity Evolution (Uncompressed)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/workdir/analyze_codes/plots_notcomp_gs/opacity_evolution.png", dpi=300)
# plt.close()

# # ============================================
# # 4️⃣ Frame별 Gaussian 생성량
# # ============================================
# plt.figure(figsize=(7,4))
# plt.bar(df["Frame_ID"], df["Num_New"], color='tab:purple')
# plt.xlabel("Frame_ID")
# plt.ylabel("Num_New")
# plt.title("New Gaussians per Frame (Uncompressed)")
# plt.tight_layout()
# plt.savefig("/workdir/analyze_codes/plots_notcomp_gs/num_new_per_frame.png", dpi=300)
# plt.close()

# # ============================================
# # 5️⃣ Opacity vs Scale 상관관계
# # ============================================
# plt.figure(figsize=(6,5))
# plt.scatter(df["Mean_Scale"], df["Mean_Opacity"], alpha=0.6, color='tab:red')
# plt.xlabel("Mean_Scale")
# plt.ylabel("Mean_Opacity")
# plt.title("Opacity vs Scale (Uncompressed)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/workdir/analyze_codes/plots_notcomp_gs/opacity_vs_scale.png", dpi=300)
# plt.close()

# print("✅ 모든 그래프가 '/workdir/analyze_codes/plots_notcomp_gs/' 폴더에 저장되었습니다.")
