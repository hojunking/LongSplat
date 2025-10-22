import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 경로 설정 ===
csv_path = "/workdir/outputs/free/grass_qp37/gaussian_track_grass.csv"
save_dir = os.path.join(os.path.dirname(csv_path), "analysis_results")
os.makedirs(save_dir, exist_ok=True)

# === 데이터 불러오기 ===
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("frame_id").reset_index(drop=True)

# frame_type 문자열 정리
df["frame_type"] = df["frame_type"].astype(str).str.strip()  # 공백 제거
df["frame_key"] = df["frame_type"].str[0]                   # 첫 글자만 추출 (I, P, B, b 등)

# === Δnum_gaussians 계산 ===
df["delta_gauss"] = df["num_gaussians"].diff().fillna(0)

# === 기본 통계 요약 ===
summary_stats = df.describe().T
summary_stats.to_csv(os.path.join(save_dir, "summary_stats.csv"))
print("✅ 기본 통계 저장 완료:", os.path.join(save_dir, "summary_stats.csv"))

# === 프레임 타입별 통계 ===
frame_summary = df.groupby("frame_type").agg({
    "num_gaussians": ["mean", "std", "min", "max"],
    "delta_gauss": ["mean", "std"],
    "small_scale_ratio": ["mean", "std"]
})
frame_summary.to_csv(os.path.join(save_dir, "frame_type_summary.csv"))
print("✅ 프레임 타입별 통계 저장 완료:", os.path.join(save_dir, "frame_type_summary.csv"))

# === 색상 매핑 ===
type_colors = {
    "I": "#ff0000",  # blue
    "P": "#ff7f0e",  # orange
    "B": "#2ca02c",  # green
    "b": "#9467bd"   # purple
}

# === 그래프 ① : 프레임별 새로 추가된 Gaussian 수 ===
plt.figure(figsize=(10,5))
for t in df["frame_type"].unique():
    subset = df[df["frame_type"] == t]
    plt.bar(
        subset["frame_id"],
        subset["delta_gauss"],
        label=f"{t}-frame",
        alpha=0.75,
        color=type_colors.get(t.strip()[0], "gray")
    )

plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Frame ID")
plt.ylabel("Δ Num Gaussians (Newly Added)")
plt.title("Incremental Gaussian Growth per Frame")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "delta_gaussian_growth.png"), dpi=300)
plt.close()

# === 그래프 ② : 타입별 Gaussian 변화 분포 (Boxplot) ===
plt.figure(figsize=(7,5))
sns.boxplot(
    data=df,
    x="frame_type",
    y="delta_gauss",
    hue="frame_key",        # 첫 글자로 색상 지정
    palette=type_colors,
    dodge=False,
    legend=False
)
plt.title("Distribution of Gaussian Growth by Frame Type")
plt.xlabel("Frame Type")
plt.ylabel("Δ Num Gaussians")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "delta_gauss_boxplot.png"), dpi=300)
plt.close()

# === 그래프 ③ : 증가량 vs 엣지비율 ===
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="delta_gauss",
    y="small_scale_ratio",
    hue="frame_key",
    palette=type_colors,
    s=40,
    alpha=0.8
)
plt.title("Small-scale Ratio vs Δ Num Gaussians")
plt.xlabel("Δ Num Gaussians (Growth)")
plt.ylabel("Small-scale Ratio (σ < 0.01)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "edge_ratio_vs_delta.png"), dpi=300)
plt.close()

# === 상관계수 분석 ===
corr = df[["num_gaussians", "delta_gauss", "small_scale_ratio"]].corr()
print("\n📊 상관계수:")
print(corr.round(3))
corr.to_csv(os.path.join(save_dir, "correlation_matrix.csv"))

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation between Gaussian Stats")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

print(f"\n✅ 분석 완료 및 그래프 저장: {save_dir}")
