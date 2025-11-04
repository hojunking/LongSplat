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
# 1️⃣ Preprocess
# ============================================
pdf.columns = pdf.columns.str.strip().str.replace(" ", "_")
pdf.rename(columns={
    "Encode_Order": "Frame_ID",
    "Y_PSNR": "PSNR_Y",
    "Frame_Type": "Type"
}, inplace=True)
pdf["Type"] = pdf["Type"].astype(str).str.strip()

# ============================================
# 2️⃣ Identify missing / skipped frames
# ============================================
used_frames = set(gdf["Frame_ID"].unique())
all_frames = set(pdf["Frame_ID"].unique())
missing_frames = sorted(list(all_frames - used_frames))

print(f"✅ Total frames in compressed log: {len(all_frames)}")
print(f"✅ Frames used in Gaussian log: {len(used_frames)}")
print(f"⚠️ Missing / skipped frames: {len(missing_frames)}")

# ============================================
# 3️⃣ Extract info for missing frames
# ============================================
missing_info = pdf[pdf["Frame_ID"].isin(missing_frames)]
used_info = pdf[pdf["Frame_ID"].isin(used_frames)]

# Save to CSV
os.makedirs("/workdir/analyze_codes/plots", exist_ok=True)
missing_info.to_csv("/workdir/analyze_codes/plots/skipped_frame_info.csv", index=False)

# ============================================
# 4️⃣ Compare statistics between used vs skipped
# ============================================
compare = pd.DataFrame({
    "Metric": ["QP", "Y_PSNR", "Bits"],
    "All_Mean": [pdf["QP"].mean(), pdf["PSNR_Y"].mean(), pdf["Bits"].mean()],
    "Used_Mean": [used_info["QP"].mean(), used_info["PSNR_Y"].mean(), used_info["Bits"].mean()],
    "Skipped_Mean": [missing_info["QP"].mean(), missing_info["PSNR_Y"].mean(), missing_info["Bits"].mean()]
})
compare["Δ (Skipped - Used)"] = compare["Skipped_Mean"] - compare["Used_Mean"]
compare["Change (%)"] = 100 * compare["Δ (Skipped - Used)"] / compare["Used_Mean"]
compare.to_csv("/workdir/analyze_codes/plots/skipped_vs_used_summary.csv", index=False)

print("📊 Comparison Summary:")
print(compare)

# ============================================
# 5️⃣ Frame type distribution of skipped frames
# ============================================
dist = missing_info["Type"].value_counts(normalize=True).reset_index()
dist.columns = ["Frame_Type", "Ratio"]
dist.to_csv("/workdir/analyze_codes/plots/skipped_frame_type_ratio.csv", index=False)
print("\n🔍 Skipped frame type distribution:")
print(dist)

# ============================================
# 6️⃣ Visualization
# ============================================

palette = {
    "I-SLICE": "#E74C3C",  # red
    "P-SLICE": "#3498DB",  # blue
    "B-SLICE": "#2ECC71",  # green
    "i-SLICE": "#E74C3C",
    "p-SLICE": "#3498DB",
    "b-SLICE": "#2ECC71"
}

# ---- PSNR distribution with skipped frames marked
plt.figure(figsize=(8,4))
sns.scatterplot(data=pdf, x="Frame_ID", y="PSNR_Y", hue="Type", alpha=0.5, palette=palette)
plt.scatter(missing_info["Frame_ID"], missing_info["PSNR_Y"],
            color="black", marker="x", s=60, label="Skipped Frame")
plt.title("PSNR Distribution (Skipped Frames Highlighted)")
plt.xlabel("Frame ID")
plt.ylabel("Y_PSNR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/skipped_frame_psnr.png", dpi=300)
plt.close()

# ---- Bits vs Frame_ID
plt.figure(figsize=(8,4))
sns.lineplot(data=pdf, x="Frame_ID", y="Bits", hue="Type", palette=palette)
plt.scatter(missing_info["Frame_ID"], missing_info["Bits"],
            color="black", marker="x", s=50, label="Skipped Frame")
plt.title("Bitrate Trend by Frame (Skipped Highlighted)")
plt.xlabel("Frame ID")
plt.ylabel("Bits")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/skipped_frame_bits.png", dpi=300)
plt.close()

# ---- PSNR vs QP scatter (Skipped vs Used)
plt.figure(figsize=(6,5))
sns.scatterplot(data=used_info, x="QP", y="PSNR_Y", color="gray", alpha=0.5, label="Used Frame")
sns.scatterplot(data=missing_info, x="QP", y="PSNR_Y", color="red", marker="x", s=70, label="Skipped Frame")
plt.title("PSNR vs QP (Skipped vs Used)")
plt.xlabel("QP")
plt.ylabel("PSNR (Y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/workdir/analyze_codes/plots/psnr_vs_qp_skipped.png", dpi=300)
plt.close()

print("✅ All skipped frame analyses saved in '/workdir/analyze_codes/plots'")
