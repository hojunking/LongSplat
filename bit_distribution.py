import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_qp_bits_per_slice(qp_csv, save_fig=False):
    """
    📊 [Compression-Aware] 슬라이스별 QP & Bit 분포 분석
    - 각 프레임 타입(Type)별 평균 QP, 평균 Bits 출력
    - Bits 분포 히스토그램 시각화

    Args:
        qp_csv (str): QP 로그 CSV 경로
        save_fig (bool): 히스토그램을 PNG로 저장 여부
    """
    if not os.path.exists(qp_csv):
        raise FileNotFoundError(f"❌ CSV not found: {qp_csv}")

    # 1️⃣ CSV 로드
    df = pd.read_csv(qp_csv)

    # 2️⃣ 필요한 컬럼 확인
    if not {"Type", "QP", "Bits"}.issubset(df.columns):
        raise KeyError("❌ 'Type', 'QP', 'Bits' 컬럼이 필요합니다.")

    # 3️⃣ 슬라이스별 QP & Bit 평균 계산
    stats = (
        df.groupby("Type")[["QP", "Bits"]]
        .agg(["mean", "std", "min", "max", "count"])
        .round(1)
    )

    # 4️⃣ 출력
    print("=== [QP & Bit Distribution per Slice Type] ===")
    print(stats.to_string())
    print("\n")

    # 5️⃣ 각 Type별 Bits 히스토그램
    plt.figure(figsize=(8, 5))
    for t in df["Type"].unique():
        subset = df[df["Type"] == t]["Bits"]
        plt.hist(subset, bins=40, alpha=0.5, label=f"{t.strip()} ({len(subset)})")

    plt.title("Bits Distribution by Slice Type")
    plt.xlabel("Bits per Frame")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_fig:
        fig_path = os.path.splitext(qp_csv)[0] + "_qp_bit_distribution.png"
        plt.savefig(fig_path, dpi=300)
        print(f"✅ Figure saved to: {fig_path}")

    plt.show()


# 실행 예시
analyze_qp_bits_per_slice("./comp_log/grass_qp37_trustmap.csv", save_fig=True)
