import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 파일 경로 ===
qp_csv = "/workdir/comp_log/grass_qp37_trustmap.csv"
inlier_csv = "/workdir/outputs/free_ema/grass_qp37_compgs_mom095_dmu05/keypoint_match_log.csv"
save_path = "/workdir/outputs/inlier_ratio_psnr_gap_final.png"

# ================================
# 1. Load compress CSV (PSNR)
# ================================
df_qp = pd.read_csv(qp_csv)
df_qp.rename(columns={"Global_Frame_ID": "Frame_ID"}, inplace=True)
df_qp = df_qp.sort_values("Frame_ID")
psnr_dict = df_qp.set_index("Frame_ID")["Y_PSNR"].to_dict()

# ================================
# 2. Load inlier CSV (Ratio)
# ================================
df_inlier = pd.read_csv(inlier_csv)
df_inlier = df_inlier.sort_values("Frame_ID")
df_inlier["inlier_ratio"] = df_inlier["Num_Inliers"] / df_inlier["Num_Keypoints"]

# ================================
# 3. Compute intersection frames
# ================================
global_ids = set(df_qp["Frame_ID"].tolist())
inlier_ids = set(df_inlier["Frame_ID"].tolist())
intersection_frames = sorted(list(global_ids.intersection(inlier_ids)))

# ================================
# 4. Compute aligned values
# ================================
inlier_x = []
inlier_y = []
psnr_x = []
psnr_y = []

for fid in intersection_frames:
    row = df_inlier[df_inlier["Frame_ID"] == fid].iloc[0]
    curr_fid = int(row["Frame_ID"])
    prev_fid = int(row["Prev_Frame_ID"])

    if curr_fid not in psnr_dict or prev_fid not in psnr_dict:
        continue

    # Inlier ratio → current frame
    inlier_x.append(curr_fid)
    inlier_y.append(float(row["inlier_ratio"]))

    # PSNR gap → curr-1 (shifted left)
    psnr_gap = abs(psnr_dict[curr_fid] - psnr_dict[prev_fid])
    psnr_x.append(curr_fid - 1)
    psnr_y.append(psnr_gap)

# ================================
# 5. Window: 160~200 범위만 시각화
# ================================
WINDOW_MIN = 160
WINDOW_MAX = 200

mask_inlier = [(WINDOW_MIN <= x <= WINDOW_MAX) for x in inlier_x]
inlier_x = np.array(inlier_x)[mask_inlier]
inlier_y = np.array(inlier_y)[mask_inlier]

mask_psnr = [(WINDOW_MIN <= x <= WINDOW_MAX) for x in psnr_x]
psnr_x = np.array(psnr_x)[mask_psnr]
psnr_y = np.array(psnr_y)[mask_psnr]

# ================================
# 6. 모든 정수 x값에 대한 데이터 준비
# ================================
# PSNR: 딕셔너리로 변환
psnr_data_dict = {int(x): y for x, y in zip(psnr_x, psnr_y)}

# Inlier: 딕셔너리로 변환
inlier_data_dict = {int(x): y for x, y in zip(inlier_x, inlier_y)}

# 원본 프레임 ID를 정렬하여 0부터 시작하는 인덱스로 매핑
original_frame_ids = sorted(set(list(psnr_data_dict.keys()) + list(inlier_data_dict.keys())))
frame_id_to_index = {fid: idx for idx, fid in enumerate(original_frame_ids)}

# 0부터 시작하는 인덱스로 변환
psnr_plot_x = []
psnr_plot_y = []
inlier_plot_x = []
inlier_plot_y = []

for original_fid in original_frame_ids:
    new_x = frame_id_to_index[original_fid]
    
    if original_fid in psnr_data_dict:
        psnr_plot_x.append(new_x)
        psnr_plot_y.append(psnr_data_dict[original_fid])
    
    if original_fid in inlier_data_dict:
        inlier_plot_x.append(new_x)
        inlier_plot_y.append(inlier_data_dict[original_fid])

# ================================
# 7. Plot (Dual Axis) - 모든 점 표시
# ================================
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
})

fig, ax1 = plt.subplots(figsize=(17, 7))

# 🔵 PSNR Gap - 선 + 모든 점
ax1.plot(psnr_plot_x, psnr_plot_y,
         color="royalblue", linewidth=3, marker="o", markersize=6, 
         markerfacecolor="royalblue", markeredgecolor="navy", markeredgewidth=1)
ax1.tick_params(axis='y', labelcolor="royalblue", labelsize=18)
ax1.grid(alpha=0.3, linestyle='--')

# 🟢 Inlier Ratio - 선 + 모든 점
ax2 = ax1.twinx()
ax2.plot(inlier_plot_x, inlier_plot_y,
         color="green", linestyle="--", linewidth=3, marker="s", markersize=6,
         markerfacecolor="green", markeredgecolor="darkgreen", markeredgewidth=1)
ax2.tick_params(axis='y', labelcolor="green", labelsize=18)

# === X축 숫자 표시 (10마다) ===
all_x_indices = list(range(len(original_frame_ids)))
x_ticks = [i for i in all_x_indices if i % 10 == 0]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks, rotation=0, fontsize=18)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"[Saved] {save_path}")
print(f"PSNR points plotted: {len(psnr_plot_x)}")
print(f"Inlier points plotted: {len(inlier_plot_x)}")