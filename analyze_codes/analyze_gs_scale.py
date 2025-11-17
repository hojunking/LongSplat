#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import math

# ============================================
# 0️⃣ Argument 설정
# ============================================
parser = argparse.ArgumentParser(
    description="Compare frame-wise Gaussian scale between compressed and uncompressed inputs with I-frame markers."
)
parser.add_argument("--compressed", required=True, help="Path to compressed gaussian_stats_log.csv")
parser.add_argument("--uncompressed", required=True, help="Path to uncompressed gaussian_stats_log.csv")
parser.add_argument("--iframe", required=True, help="Path to x265 CSV containing Frame_Type info")
parser.add_argument("--output", default="/workdir/analyze_codes/plots", help="Directory to save results")
args = parser.parse_args()

compressed_csv = args.compressed
uncompressed_csv = args.uncompressed
iframe_csv = args.iframe
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

# ============================================
# 1️⃣ 데이터 로드 함수
# ============================================
def load_gaussian_stats(path):
    df = pd.read_csv(path)
    if not {"Frame_ID", "Mean_Scale", "Std_Scale"}.issubset(df.columns):
        raise ValueError(f"❌ {path} must contain 'Frame_ID', 'Mean_Scale', 'Std_Scale'")
    df = (
        df.groupby("Frame_ID")[["Mean_Scale", "Std_Scale"]]
        .mean()
        .reset_index()
        .sort_values("Frame_ID")
    )
    print(f"✅ Loaded {len(df)} entries from {path}")
    return df

compressed = load_gaussian_stats(compressed_csv)
uncompressed = load_gaussian_stats(uncompressed_csv)

# ============================================
# 2️⃣ I-frame CSV 로드 및 매핑
# ============================================
meta = pd.read_csv(iframe_csv)
meta.columns = meta.columns.str.strip().str.replace(" ", "_")

frame_col = next((c for c in meta.columns if "Encode" in c or "Frame" in c), None)
type_col = next((c for c in meta.columns if "Type" in c), None)

if frame_col is None or type_col is None:
    raise ValueError(f"❌ Could not find suitable columns for Frame_ID/Type in {iframe_csv}")

meta.rename(columns={frame_col: "Frame_ID", type_col: "Type"}, inplace=True)
meta["Type"] = meta["Type"].astype(str).str.strip()

iframe_ids_raw = meta.loc[meta["Type"].str.upper().eq("I-SLICE"), "Frame_ID"].tolist()

def x265_to_gaussian_id(fid):
    skipped = math.floor(fid / 9) + 1
    mapped = fid - skipped
    return mapped if mapped >= 0 else None

mapped_iframes = [x265_to_gaussian_id(fid) for fid in iframe_ids_raw]
iframe_ids = [fid for fid in mapped_iframes if fid in compressed["Frame_ID"].values]
print(f"📍 Found {len(iframe_ids)} aligned I-frames (mapped to Gaussian index): {iframe_ids}")

# ============================================
# 3️⃣ 비교 결과 계산
# ============================================
merged = pd.merge(
    compressed, uncompressed, on="Frame_ID", suffixes=("_compressed", "_uncompressed"), how="inner"
)
merged["Δ_Mean_Scale"] = merged["Mean_Scale_compressed"] - merged["Mean_Scale_uncompressed"]
merged["Δ_Std_Scale"] = merged["Std_Scale_compressed"] - merged["Std_Scale_uncompressed"]

summary = {
    "Mean_Scale_compressed_mean": merged["Mean_Scale_compressed"].mean(),
    "Mean_Scale_uncompressed_mean": merged["Mean_Scale_uncompressed"].mean(),
    "Δ_Mean_Scale_avg": merged["Δ_Mean_Scale"].mean(),
    "Std_Scale_compressed_mean": merged["Std_Scale_compressed"].mean(),
    "Std_Scale_uncompressed_mean": merged["Std_Scale_uncompressed"].mean(),
    "Δ_Std_Scale_avg": merged["Δ_Std_Scale"].mean(),
}

summary_df = pd.DataFrame([summary])
comparison_csv = os.path.join(output_dir, "comparison_scale_summary.csv")
merged.to_csv(os.path.join(output_dir, "framewise_scale_comparison.csv"), index=False)
summary_df.to_csv(comparison_csv, index=False)

print("✅ Saved comparison results:")
print(f" - {comparison_csv}")
print(summary_df.T)

# ============================================
# 4️⃣ 그래프 그리기
# ============================================
plt.figure(figsize=(10, 5))
plt.plot(uncompressed["Frame_ID"], uncompressed["Mean_Scale"], label="Uncompressed", color="tab:blue")
plt.plot(compressed["Frame_ID"], compressed["Mean_Scale"], label="Compressed (QP)", color="tab:green", linestyle="--")
plt.fill_between(
    compressed["Frame_ID"],
    compressed["Mean_Scale"] - compressed["Std_Scale"],
    compressed["Mean_Scale"] + compressed["Std_Scale"],
    alpha=0.2, color="tab:green"
)
plt.fill_between(
    uncompressed["Frame_ID"],
    uncompressed["Mean_Scale"] - uncompressed["Std_Scale"],
    uncompressed["Mean_Scale"] + uncompressed["Std_Scale"],
    alpha=0.2, color="tab:blue"
)

# 🔴 I-frame 표시
for fid in iframe_ids:
    plt.axvline(x=fid, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

plt.title("Gaussian Mean Scale Comparison (Compressed vs Uncompressed)")
plt.xlabel("Frame ID (Gaussian)")
plt.ylabel("Mean Gaussian Scale")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

scene_name = os.path.basename(os.path.dirname(compressed_csv))
output_path = os.path.join(output_dir, f"compare_gaussian_scale_{scene_name}.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Saved plot: {output_path}")
