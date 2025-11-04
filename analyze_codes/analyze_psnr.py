import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ CSV 파일 로드
csv_path = "./comp_log/x265_3dgs-dataset__free_dataset__free_dataset__grass__images_qp32.csv"
df = pd.read_csv(csv_path)

psnr_col = " Y PSNR"
frame_col = "Encode Order"

# 2️⃣ usable frame list 만들기 (init 1~3 제외, 9의 배수 제외)
valid_frames = [f for f in df[frame_col] if (f >= 3) and (f % 9 != 0)]

records = []
prev = None
for curr in valid_frames:
    psnr_curr = df.loc[df[frame_col] == curr, psnr_col].values[0]
    if prev is None:
        records.append([curr, None, 0.0, 0.0])
    else:
        psnr_prev = df.loc[df[frame_col] == prev, psnr_col].values[0]
        delta = psnr_curr - psnr_prev
        abs_delta = abs(delta)
        records.append([curr, prev, delta, abs_delta])
    prev = curr

# 3️⃣ DataFrame 생성
df_delta = pd.DataFrame(records, columns=["Frame_ID", "Prev_Frame_ID", "Delta_PSNR_Y", "Abs_Delta_PSNR_Y"])

# 4️⃣ 그래프 시각화 (1) 원래 ΔPSNR (부호 있음)
plt.figure(figsize=(10, 4))
plt.plot(df_delta["Frame_ID"], df_delta["Delta_PSNR_Y"], color="royalblue", linewidth=1.5)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Grass QP37 - ΔPSNR(Y) vs Frame (Valid train frames only)", fontsize=13)
plt.xlabel("Frame ID", fontsize=11)
plt.ylabel("ΔPSNR (Y)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("grass_qp32_delta_psnrY_correct.png", dpi=300, bbox_inches="tight")
plt.close()

# 5️⃣ 그래프 시각화 (2) 절댓값 |ΔPSNR|
plt.figure(figsize=(10, 4))
plt.plot(df_delta["Frame_ID"], df_delta["Abs_Delta_PSNR_Y"], color="darkorange", linewidth=1.5)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Grass QP37 - |ΔPSNR(Y)| vs Frame (Valid train frames only)", fontsize=13)
plt.xlabel("Frame ID", fontsize=11)
plt.ylabel("|ΔPSNR (Y)|", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("grass_qp37_abs_delta_psnrY_correct.png", dpi=300, bbox_inches="tight")
plt.close()

# 6️⃣ CSV 저장
df_delta.to_csv("grass_qp32_delta_psnrY_correct.csv", index=False)

print("✅ ΔPSNR (Y) CSV 및 그래프(원본, 절댓값) 저장 완료!")
