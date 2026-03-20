#!/usr/bin/env python3
import pandas as pd
import os
import glob
import re

def build_global_frame_table(qp_csv, output_csv, gop_len=32, debug=False):
    # 1️⃣ Load encoder log
    enc = pd.read_csv(qp_csv, sep='\t|,', engine='python')
    enc.columns = enc.columns.str.strip().str.replace(" ", "_")

    # 2️⃣ I-frame (GOP 경계) 찾기
    i_indices = enc.index[enc["Type"].str.strip().str.upper() == "I-SLICE"].tolist()
    if len(i_indices) == 0:
        raise ValueError(f"No I-frames found in {qp_csv}")

    # 3️⃣ GOP index 부여
    gop_idx = []
    for i in range(len(enc)):
        current_gop = sum([i >= idx for idx in i_indices]) - 1
        gop_idx.append(current_gop)

    enc["GOP_Index"] = gop_idx

    # 4️⃣ GOP 내에서 POC 순서 정렬
    enc = enc.sort_values(by=["GOP_Index", "POC"]).reset_index(drop=True)

    # 5️⃣ Global Frame ID 계산
    enc["Global_Frame_ID"] = enc["GOP_Index"] * gop_len + enc["POC"]

    # 6️⃣ Test frame 설정
    enc["Is_Test"] = enc["Global_Frame_ID"].apply(lambda x: x % 9 == 0)

    # 7️⃣ 열 순서 정리
    cols = ["GOP_Index", "Global_Frame_ID", "Is_Test"] + [
        c for c in enc.columns if c not in ["GOP_Index", "Global_Frame_ID", "Is_Test"]
    ]
    enc = enc[cols]

    # 8️⃣ 저장
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    enc.to_csv(output_csv, index=False)

    print(f"✅ Saved: {output_csv}")

    if debug:
        print(enc.head())


# ============================================
# 🎬 모든 CSV 자동 처리
# ============================================
if __name__ == "__main__":

    BASE_DIR = "/workdir/comp_log"

    csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))

    for qp_csv in csv_files:

        filename = os.path.basename(qp_csv)

        # scene 추출
        scene_match = re.search(r"free_dataset_free_dataset_(.*?)_images", filename)

        # qp 추출
        qp_match = re.search(r"(qp\d+)", filename, re.IGNORECASE)

        if scene_match is None or qp_match is None:
            print(f"⚠️ Skip (cannot parse name): {filename}")
            continue

        scene = scene_match.group(1)
        qp = qp_match.group(1).lower()

        output_csv = os.path.join(BASE_DIR, f"{scene}_{qp}_vvc_trustmap.csv")

        print("\n====================================")
        print(f"📂 Processing: {filename}")
        print(f"Scene: {scene}")
        print(f"QP: {qp}")
        print(f"Output: {output_csv}")
        print("====================================")

        try:
            build_global_frame_table(qp_csv, output_csv)
        except Exception as e:
            print(f"❌ Error: {e}")