import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1️⃣ 파일 경로 설정
# -------------------------------------------------------
psnr_csv = "./grass_qp37_delta_psnrY_correct.csv"      # ΔPSNR 데이터
keypoint_csv = "/workdir/outputs/free/grass_qp37/keypoint_match_log.csv"              # keypoint/inlier 데이터

# -------------------------------------------------------
# 2️⃣ CSV 로드 및 전처리
# -------------------------------------------------------
psnr_df = pd.read_csv(psnr_csv)
kp_df = pd.read_csv(keypoint_csv)

# 열 이름 정리
psnr_df.columns = [c.strip() for c in psnr_df.columns]
kp_df.columns = [c.strip() for c in kp_df.columns]

# 공통 열인 Frame_ID 기준 병합
merged = pd.merge(kp_df, psnr_df, on="Frame_ID", how="inner")

# inlier 비율 및 감소 비율 계산
merged["Inlier_Ratio"] = merged["Num_Inliers"] / merged["Num_Keypoints"]
merged["Inlier_Reduction"] = 1 - merged["Inlier_Ratio"]

# PSNR 하락 프레임만 따로 추출
drop_df = merged[merged["Delta_PSNR_Y"] < 0]

print(f"✅ 총 {len(merged)}개 중 PSNR 하락 프레임 {len(drop_df)}개")

# -------------------------------------------------------
# 3️⃣ 상관계수 계산
# -------------------------------------------------------
corr_all = merged[["Delta_PSNR_Y", "Num_Keypoints", "Num_Inliers", "Inlier_Ratio"]].corr()
corr_drop = drop_df[["Delta_PSNR_Y", "Num_Keypoints", "Num_Inliers", "Inlier_Ratio"]].corr()

print("\n[전체 프레임 상관계수]")
print(corr_all.round(3))
print("\n[PSNR 하락 프레임 상관계수]")
print(corr_drop.round(3))

# -------------------------------------------------------
# 4️⃣ 시각화 (산점도 + 상관 히트맵)
# -------------------------------------------------------
plt.figure(figsize=(12, 5))

# (1) ΔPSNR vs Inlier Ratio
plt.subplot(1, 2, 1)
plt.scatter(merged["Delta_PSNR_Y"], merged["Inlier_Ratio"], c="royalblue", alpha=0.7)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title("ΔPSNR vs Inlier Ratio (All Frames)")
plt.xlabel("ΔPSNR (Y)")
plt.ylabel("Inlier Ratio")
plt.grid(True, linestyle="--", alpha=0.5)

# (2) PSNR 하락 프레임만
plt.subplot(1, 2, 2)
plt.scatter(drop_df["Delta_PSNR_Y"], drop_df["Inlier_Ratio"], c="darkorange", alpha=0.7)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title("ΔPSNR vs Inlier Ratio (PSNR Drop Frames)")
plt.xlabel("ΔPSNR (Y)")
plt.ylabel("Inlier Ratio")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("psnr_inlier_scatter.png", dpi=300, bbox_inches="tight")
plt.close()

# (3) 상관계수 히트맵
plt.figure(figsize=(7, 5))
sns.heatmap(corr_all, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (All Frames)")
plt.tight_layout()
plt.savefig("psnr_inlier_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("📊 그래프 저장 완료: psnr_inlier_scatter.png / psnr_inlier_heatmap.png")

# -------------------------------------------------------
# 5️⃣ 결과 CSV 저장
# -------------------------------------------------------
merged.to_csv("merged_psnr_inlier.csv", index=False)
print("💾 merged_psnr_inlier.csv 저장 완료.")
