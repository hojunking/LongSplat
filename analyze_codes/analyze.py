import pandas as pd
import numpy as np
import os

# ============================================
# Load compressed & uncompressed logs
# ============================================
uncompressed   = pd.read_csv("/workdir/outputs/free/grass_debug_2/gaussian_stats_log.csv")
compressed   = pd.read_csv("/workdir/outputs/free/grass_qp37_debug_2/gaussian_stats_log.csv")

# Frame-wise 평균 계산
def summarize(df):
    return {
        "Total_Iterations": df["Iteration"].max(),
        "Total_Frames": df["Frame_ID"].nunique(),
        "Final_Num_Gaussians": df["Num_Gaussians"].iloc[-1],
        "Mean_Num_New": df["Num_New"].mean(),
        "Mean_Scale": df["Mean_Scale"].mean(),
        "Std_Scale": df["Mean_Scale"].std(),
        "Mean_Opacity": df["Mean_Opacity"].mean(),
        "Std_Opacity": df["Mean_Opacity"].std(),
        "Scale_Opacity_Corr": df["Mean_Scale"].corr(df["Mean_Opacity"])
    }

un = summarize(uncompressed)
cp = summarize(compressed)

# ============================================
# 1️⃣ 비교 요약
# ============================================
comparison = pd.DataFrame({
    "Metric": list(un.keys()),
    "Uncompressed": list(un.values()),
    "Compressed": list(cp.values()),
})
comparison["Δ (Compressed - Uncompressed)"] = comparison["Compressed"] - comparison["Uncompressed"]
comparison["Change (%)"] = (
    (comparison["Compressed"] - comparison["Uncompressed"]) / comparison["Uncompressed"] * 100
).replace([np.inf, -np.inf], np.nan)

# ============================================
# 2️⃣ 출력 및 저장
# ============================================
os.makedirs("/workdir/analyze_codes/plots", exist_ok=True)
comparison.to_csv("/workdir/analyze_codes/plots/comparison_summary_grass.csv", index=False)

print("✅ Saved comparison summary to '/workdir/analyze_codes/plots/comparison_summary.csv'")
print(comparison.round(5))
