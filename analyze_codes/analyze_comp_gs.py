import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# 0️⃣ Load data
# ============================================
gdf = pd.read_csv("/workdir/outputs/free/grass_qp37_debug_2/gaussian_stats_log.csv")
pdf = pd.read_csv("/workdir/comp_log/x265_3dgs-dataset__free_dataset__free_dataset__grass__images_qp37.csv")

# ============================================
# 1️⃣ Preprocess columns
# ============================================
pdf.columns = pdf.columns.str.strip().str.replace(" ", "_")
pdf.rename(columns={
    "Encode_Order": "Frame_ID",
    "Y_PSNR": "PSNR_Y",
    "Frame_Type": "Type"
}, inplace=True)
pdf["Type"] = pdf["Type"].astype(str).str.strip()

# ============================================
# 2️⃣ Merge two CSVs
# ============================================
merged = pd.merge(gdf, pdf, on="Frame_ID", how="inner")
print("✅ Merged shape:", merged.shape)

# ============================================
# 3️⃣ Correlation analysis
# ============================================
corr = merged[["PSNR_Y", "QP", "Num_Gaussians", "Num_New", "Mean_Opacity", "Mean_Scale"]].corr()
print("📈 Correlation Matrix:\n", corr)

# ============================================
# 4️⃣ Statistical summary per Frame Type
# ============================================
summary_type = (
    merged.groupby("Type")[["PSNR_Y", "Num_Gaussians", "Num_New", "Mean_Scale", "Mean_Opacity"]]
    .agg(["mean", "std", "min", "max"])
)
summary_type.columns = ["_".join(col).strip() for col in summary_type.columns.values]
summary_type.reset_index(inplace=True)

# ============================================
# 5️⃣ Global numerical summary
# ============================================
summary_global = {
    "Total_Frames": merged["Frame_ID"].nunique(),
    "Mean_PSNR": merged["PSNR_Y"].mean(),
    "Std_PSNR": merged["PSNR_Y"].std(),
    "Mean_Num_Gaussians": merged["Num_Gaussians"].mean(),
    "Mean_Num_New": merged["Num_New"].mean(),
    "Mean_Scale": merged["Mean_Scale"].mean(),
    "Std_Scale": merged["Mean_Scale"].std(),
    "Mean_Opacity": merged["Mean_Opacity"].mean(),
    "Std_Opacity": merged["Mean_Opacity"].std(),
    "Scale_Opacity_Corr": merged["Mean_Scale"].corr(merged["Mean_Opacity"]),
    "PSNR_Scale_Corr": merged["PSNR_Y"].corr(merged["Mean_Scale"]),
    "PSNR_Gaussian_Corr": merged["PSNR_Y"].corr(merged["Num_Gaussians"]),
}
summary_global_df = pd.DataFrame([summary_global])

# ============================================
# 6️⃣ Save all numeric results
# ============================================
os.makedirs("/workdir/analyze_codes/plots", exist_ok=True)
summary_type.to_csv("/workdir/analyze_codes/plots/stat_summary_per_type.csv", index=False)
summary_global_df.to_csv("/workdir/analyze_codes/plots/stat_summary_global.csv", index=False)
corr.to_csv("/workdir/analyze_codes/plots/correlation_matrix.csv")

print("✅ Saved:")
print(" - stat_summary_per_type.csv (Frame Type별 평균/표준편차)")
print(" - stat_summary_global.csv (전체 요약)")
print(" - correlation_matrix.csv (전체 상관계수)")

# ============================================
# 7️⃣ Visualization (색상 통일)
# ============================================

palette = {
    "I-SLICE": "#E74C3C",  # 🔴 Red
    "P-SLICE": "#3498DB",  # 🔵 Blue
    "B-SLICE": "#2ECC71",  # 🟢 Green
    "i-SLICE": "#E74C3C",
    "p-SLICE": "#3498DB",
    "b-SLICE": "#2ECC71"
}

# ---- ① PSNR vs Num_Gaussians ----
plt.figure(figsize=(6,4))
sns.scatterplot(data=merged, x="PSNR_Y", y="Num_Gaussians", hue="Type", palette=palette)
plt.title("PSNR vs Number of Gaussians by Frame Type")
plt.xlabel("PSNR (Y)")
plt.ylabel("Num_Gaussians")
plt.grid(True)
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/psnr_vs_num_gaussians.png", dpi=300)
plt.close()

# ---- ② QP vs Mean_Scale ----
plt.figure(figsize=(6,4))
sns.boxplot(data=merged, x="QP", y="Mean_Scale", hue="Type", palette=palette)
plt.title("Mean Scale by QP and Frame Type")
plt.xlabel("QP")
plt.ylabel("Mean Scale")
plt.legend(title="Frame Type")
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/qp_vs_mean_scale.png", dpi=300)
plt.close()

# ---- ③ PSNR vs Mean_Opacity ----
plt.figure(figsize=(6,4))
sns.regplot(data=merged, x="PSNR_Y", y="Mean_Opacity", scatter_kws={'alpha':0.5}, color="gray", line_kws={'color':'black'})
plt.title("PSNR vs Mean Opacity")
plt.xlabel("PSNR (Y)")
plt.ylabel("Mean Opacity")
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/psnr_vs_mean_opacity.png", dpi=300)
plt.close()

# ---- ④ Frame Type별 Gaussian 생성량 ----
plt.figure(figsize=(5,4))
sns.barplot(data=merged, x="Type", y="Num_New", estimator="mean", palette=palette, errorbar=None)
plt.title("Average New Gaussians per Frame Type")
plt.ylabel("Num_New (mean)")
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/num_new_per_type.png", dpi=300)
plt.close()

# ---- ⑤ PSNR vs Mean_Scale ---- ✅ 추가
plt.figure(figsize=(6,4))
sns.scatterplot(data=merged, x="PSNR_Y", y="Mean_Scale", hue="Type", palette=palette)
sns.lineplot(
    data=merged.sort_values("PSNR_Y"),
    x="PSNR_Y", y="Mean_Scale", color="black", lw=1.5, label="Trend"
)
plt.title("PSNR vs Mean Scale by Frame Type")
plt.xlabel("PSNR (Y)")
plt.ylabel("Mean_Scale")
plt.legend(title="Frame Type + Trend")
plt.grid(True)
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/psnr_vs_mean_scale.png", dpi=300)
plt.close()

print("✅ All plots and stats saved in '/workdir/analyze_codes/plots'")
