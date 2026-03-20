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
# 🎬 모든 SCENE + QP 실행
# ============================================
if __name__ == "__main__":
    #SCENES = ["grass", "hydrant", "lab", "pillar", "road", "sky", "stair"] # Free
    #SCENES = ["grass"]  # TNT
    SCENES = ["forest1", "forest2", "forest3", "garden1", "garden2", "garden3", 
              "indoor", "playground", "university1", "university2", "university3", "university4"]  # Hike

    QP_LEVELS = ["QP37"]
    
    BASE_DIR = "/workdir/comp_log/hike"
    
    for scene in SCENES:
        for qp in QP_LEVELS:
            # qp_csv = f"{BASE_DIR}/x265_3dgs-dataset__Tanks__{scene}__images_{qp}.csv"
            qp_csv = f"{BASE_DIR}/images_x265_{qp}_{scene}.csv"
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




#  SCENES=("Ballroom" "Barn" "Church" "Francis" "Horse" "Ignatius" "Museum")
# ============================================
# 🎬 단일 파일 처리
# ============================================
# if __name__ == "__main__":
#     scene = "Family"
    
    
#     qp_csv = f"temp/Tanks/{scene}/images/{scene}_x265_QP27_with_tld.csv"
#     output_csv = f"comp_log/Tanks/{scene}_qp27_trustmap.csv"
    
#     print(f"\n====================================")
#     print(f"📂 Processing single file")
#     print(f"Input:  {qp_csv}")
#     print(f"Output: {output_csv}")
#     print(f"====================================")
    
#     try:
#         build_global_frame_table(qp_csv, output_csv, debug=False)
#     except Exception as e:
#         print(f"❌ Error processing file: {e}")
    
#     qp_csv = f"temp/Tanks/{scene}/images/{scene}_x265_QP42_with_tld.csv"
#     output_csv = f"comp_log/Tanks/{scene}_qp42_trustmap.csv"
    
#     print(f"\n====================================")
#     print(f"📂 Processing single file")
#     print(f"Input:  {qp_csv}")
#     print(f"Output: {output_csv}")
#     print(f"====================================")
    
#     try:
#         build_global_frame_table(qp_csv, output_csv, debug=False)
#     except Exception as e:
#         print(f"❌ Error processing file: {e}")


#     qp_csv = f"temp/Tanks/{scene}/images/{scene}_x265_QP47_with_tld.csv"
#     output_csv = f"comp_log/Tanks/{scene}_qp47_trustmap.csv"
    
#     print(f"\n====================================")
#     print(f"📂 Processing single file")
#     print(f"Input:  {qp_csv}")
#     print(f"Output: {output_csv}")
#     print(f"====================================")
    
#     try:
#         build_global_frame_table(qp_csv, output_csv, debug=False)
#     except Exception as e:
#         print(f"❌ Error processing file: {e}")