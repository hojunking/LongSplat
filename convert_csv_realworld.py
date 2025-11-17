#!/usr/bin/env python3
import pandas as pd
import os

def build_global_frame_table(qp_csv, output_csv, gop_len=32, debug=False):
    # 1️⃣ Load x265 log
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
    
    # 4️⃣ GOP 내에서 POC 순서대로 정렬
    enc = enc.sort_values(by=["GOP_Index", "POC"]).reset_index(drop=True)
    
    # 5️⃣ Global Frame ID 계산
    enc["Global_Frame_ID"] = enc["GOP_Index"] * gop_len + enc["POC"]
    
    # 6️⃣ Is_Test 설정 (9의 배수)
    enc["Is_Test"] = enc["Global_Frame_ID"].apply(lambda x: x % 9 == 0)
    
    # 7️⃣ 열 순서 정리
    cols = ["GOP_Index", "Global_Frame_ID", "Is_Test"] + [
        col for col in enc.columns if col not in ["GOP_Index", "Global_Frame_ID", "Is_Test"]
    ]
    enc = enc[cols]
    
    # 8️⃣ 저장
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    enc.to_csv(output_csv, index=False)
    print(f"✅ Saved preprocessed CSV: {output_csv}")
    
    if debug:
        print(enc.head(10))


# ============================================
# 🎬 Real-world YouTube Dataset 실행
# ============================================
if __name__ == "__main__":
    SCENES = ["IMG_0405", "IMG_0406"]
    QP_LEVELS = ["QP37"]
    
    BASE_DIR = "/workdir/comp_log"
    
    for scene in SCENES:
        for qp in QP_LEVELS:
            qp_csv = f"{BASE_DIR}/{scene}_x265_{qp}.csv"
            output_csv = f"{BASE_DIR}/{scene}_{qp}_trustmap.csv"
            
            print(f"\n====================================")
            print(f"📂 Processing: {scene} ({qp})")
            print(f"Input:  {qp_csv}")
            print(f"Output: {output_csv}")
            print(f"====================================")
            
            try:
                build_global_frame_table(qp_csv, output_csv, debug=False)
            except Exception as e:
                print(f"❌ Error processing {scene} ({qp}): {e}")
