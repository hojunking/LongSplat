import pandas as pd
import numpy as np
import os


def determine_qp_parameter(qp_csv, method='auto', debug=False):
    """
    Scene의 bits 분포 특성에 따라 최적 QP parameter를 자동으로 결정
    
    Args:
        qp_csv (str): QP 로그 CSV 경로
        method (str): 결정 방법
            - 'auto': 자동 결정 (권장)
            - 'p_frame_ratio': P-frame 비율 기반
            - 'cv': 변동계수 기반
            - 'bit_trust_var': bit_trust 분산 기반
            - 'combined': 여러 지표 종합
        debug (bool): 디버그 출력 여부
    
    Returns:
        float: 최적 QP parameter (0.4 ~ 1.0)
    """
    
    if not os.path.exists(qp_csv):
        raise FileNotFoundError(f"❌ QP CSV not found: {qp_csv}")
    
    df = pd.read_csv(qp_csv)
    
    # Frame ID 표준화
    if "Global_Frame_ID" in df.columns:
        df = df.rename(columns={"Global_Frame_ID": "Frame_ID"})
    
    # Type 컬럼 정리
    df['Type'] = df['Type'].str.strip()
    
    # Bits 컬럼 확인
    bit_col_candidates = ["Bits", "bits", "FrameBits", "Frame_Bits"]
    bit_col = next((c for c in bit_col_candidates if c in df.columns), None)
    if bit_col is None:
        raise KeyError(f"❌ No bits column found in {qp_csv}")
    
    bits = df[bit_col].values
    total_frames = len(df)
    
    # ============================================
    # 특성 추출
    # ============================================
    
    # 1. 기본 통계
    mean_bits = np.mean(bits)
    std_bits = np.std(bits)
    cv = (std_bits / mean_bits) * 100  # 변동계수
    
    # 2. 프레임 타입별 분석
    frame_type_counts = {}
    frame_type_avg_bits = {}
    
    for frame_type in ['I-SLICE', 'P-SLICE', 'B-SLICE', 'b-SLICE']:
        mask = df['Type'] == frame_type
        count = mask.sum()
        frame_type_counts[frame_type] = count
        if count > 0:
            frame_type_avg_bits[frame_type] = df[mask][bit_col].mean()
        else:
            frame_type_avg_bits[frame_type] = 0
    
    # P-frame 비율
    p_frame_ratio = frame_type_counts.get('P-SLICE', 0) / total_frames
    
    # I-frame 비율
    i_frame_ratio = frame_type_counts.get('I-SLICE', 0) / total_frames
    
    # 3. bit_trust 계산
    min_bits = np.min(bits)
    max_bits = np.max(bits)
    bit_trust = (bits - min_bits) / (max_bits - min_bits + 1e-8) * 0.5
    
    mean_bit_trust = np.mean(bit_trust)
    std_bit_trust = np.std(bit_trust)
    
    # bit_trust 변화량
    bit_trust_diff = np.abs(np.diff(bit_trust))
    mean_bit_trust_diff = np.mean(bit_trust_diff)
    max_bit_trust_diff = np.max(bit_trust_diff)
    
    # 4. I-frame의 영향
    i_frame_mask = df['Type'] == 'I-SLICE'
    if i_frame_mask.sum() > 0:
        i_frame_bit_trust = bit_trust[i_frame_mask].mean()
        i_frame_avg = frame_type_avg_bits['I-SLICE']
    else:
        i_frame_bit_trust = 0
        i_frame_avg = 0
    
    # I-frame vs non-I-frame 비율
    non_i_frames = [ft for ft in ['P-SLICE', 'B-SLICE', 'b-SLICE'] 
                    if frame_type_avg_bits[ft] > 0]
    if non_i_frames and i_frame_avg > 0:
        non_i_avg = np.mean([frame_type_avg_bits[ft] for ft in non_i_frames])
        i_to_non_i_ratio = i_frame_avg / non_i_avg
    else:
        i_to_non_i_ratio = 1.0
    
    # ============================================
    # QP parameter 결정 로직
    # ============================================
    
    qp_param = 0.4  # 기본값
    reason = ""
    
    if method == 'p_frame_ratio':
        # 방법 1: P-frame 비율 기반
        if p_frame_ratio > 0.50:
            qp_param = 0.8
            reason = f"P-frame 비율 높음 ({p_frame_ratio*100:.1f}%)"
        elif p_frame_ratio > 0.40:
            qp_param = 0.6
            reason = f"P-frame 비율 중간 ({p_frame_ratio*100:.1f}%)"
        elif p_frame_ratio > 0.35:
            qp_param = 0.5
            reason = f"P-frame 비율 약간 높음 ({p_frame_ratio*100:.1f}%)"
        else:
            qp_param = 0.4
            reason = f"P-frame 비율 정상 ({p_frame_ratio*100:.1f}%)"
    
    elif method == 'cv':
        # 방법 2: 변동계수(CV) 기반
        if cv > 60:
            qp_param = 0.8
            reason = f"CV 매우 높음 ({cv:.1f}%)"
        elif cv > 50:
            qp_param = 0.6
            reason = f"CV 높음 ({cv:.1f}%)"
        elif cv > 40:
            qp_param = 0.5
            reason = f"CV 약간 높음 ({cv:.1f}%)"
        else:
            qp_param = 0.4
            reason = f"CV 정상 ({cv:.1f}%)"
    
    elif method == 'bit_trust_var':
        # 방법 3: bit_trust 변동성 기반
        if mean_bit_trust_diff > 0.08:
            qp_param = 0.8
            reason = f"bit_trust 변화 매우 큼 ({mean_bit_trust_diff:.4f})"
        elif mean_bit_trust_diff > 0.06:
            qp_param = 0.6
            reason = f"bit_trust 변화 큼 ({mean_bit_trust_diff:.4f})"
        elif mean_bit_trust_diff > 0.04:
            qp_param = 0.5
            reason = f"bit_trust 변화 약간 큼 ({mean_bit_trust_diff:.4f})"
        else:
            qp_param = 0.4
            reason = f"bit_trust 변화 정상 ({mean_bit_trust_diff:.4f})"
    
    elif method == 'combined' or method == 'auto':
        # 방법 4: 종합 점수 기반 (권장)
        score = 0.0
        reasons = []
        
        # 점수 1: P-frame 비율 (가중치 40%)
        if p_frame_ratio > 0.50:
            score += 0.4 * 1.0
            reasons.append(f"P-frame 비율 높음 ({p_frame_ratio*100:.1f}%)")
        elif p_frame_ratio > 0.40:
            score += 0.4 * 0.7
            reasons.append(f"P-frame 비율 중간 ({p_frame_ratio*100:.1f}%)")
        elif p_frame_ratio > 0.35:
            score += 0.4 * 0.4
            reasons.append(f"P-frame 비율 약간 높음 ({p_frame_ratio*100:.1f}%)")
        
        # 점수 2: CV (가중치 30%)
        if cv > 60:
            score += 0.3 * 1.0
            reasons.append(f"CV 매우 높음 ({cv:.1f}%)")
        elif cv > 50:
            score += 0.3 * 0.7
            reasons.append(f"CV 높음 ({cv:.1f}%)")
        elif cv > 40:
            score += 0.3 * 0.4
            reasons.append(f"CV 약간 높음 ({cv:.1f}%)")
        
        # 점수 3: bit_trust 변화량 (가중치 20%)
        if mean_bit_trust_diff > 0.08:
            score += 0.2 * 1.0
            reasons.append(f"bit_trust 변화 큼 ({mean_bit_trust_diff:.4f})")
        elif mean_bit_trust_diff > 0.06:
            score += 0.2 * 0.7
        elif mean_bit_trust_diff > 0.04:
            score += 0.2 * 0.4
        
        # 점수 4: I-frame 비율 (가중치 10%)
        if i_to_non_i_ratio < 2.0:  # I-frame이 작으면 불안정
            score += 0.1 * 0.5
            reasons.append(f"I-frame 상대적으로 작음 ({i_to_non_i_ratio:.2f}x)")
        
        # 점수에 따라 QP parameter 결정
        if score >= 0.7:
            qp_param = 1.0
            reason = "종합 점수 매우 높음"
        elif score >= 0.5:
            qp_param = 0.8
            reason = "종합 점수 높음"
        elif score >= 0.3:
            qp_param = 0.6
            reason = "종합 점수 중간"
        elif score >= 0.15:
            qp_param = 0.5
            reason = "종합 점수 약간 높음"
        else:
            qp_param = 0.4
            reason = "종합 점수 정상"
        
        reason = f"{reason} (score={score:.2f}): " + ", ".join(reasons[:2])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # ============================================
    # 디버그 출력
    # ============================================
    
    if debug:
        print("\n" + "="*80)
        print(f"🔍 QP Parameter 자동 결정 (method={method})")
        print("="*80)
        print(f"\n📊 Scene 특성:")
        print(f"  총 프레임: {total_frames}")
        print(f"  평균 Bits: {mean_bits:,.0f}")
        print(f"  CV: {cv:.2f}%")
        print(f"\n📈 프레임 타입 분포:")
        print(f"  I-frame: {frame_type_counts.get('I-SLICE', 0):3d} ({i_frame_ratio*100:5.1f}%) - 평균 {frame_type_avg_bits['I-SLICE']:,.0f} bits")
        print(f"  P-frame: {frame_type_counts.get('P-SLICE', 0):3d} ({p_frame_ratio*100:5.1f}%) - 평균 {frame_type_avg_bits['P-SLICE']:,.0f} bits")
        print(f"  B-frame: {frame_type_counts.get('B-SLICE', 0):3d} ({frame_type_counts.get('B-SLICE', 0)/total_frames*100:5.1f}%)")
        print(f"  b-frame: {frame_type_counts.get('b-SLICE', 0):3d} ({frame_type_counts.get('b-SLICE', 0)/total_frames*100:5.1f}%)")
        print(f"\n🎯 bit_trust 분석:")
        print(f"  평균: {mean_bit_trust:.3f} ± {std_bit_trust:.3f}")
        print(f"  평균 변화: {mean_bit_trust_diff:.4f} (최대: {max_bit_trust_diff:.4f})")
        print(f"  I-frame bit_trust: {i_frame_bit_trust:.3f}")
        print(f"\n📐 I-frame 영향:")
        print(f"  I/non-I 비율: {i_to_non_i_ratio:.2f}x")
        print(f"\n✅ 결정된 QP parameter: {qp_param}")
        print(f"   이유: {reason}")
        print("="*80)
    
    return qp_param


# ============================================
# 사용 예시 및 테스트
# ============================================

if __name__ == "__main__":
    SCENES = ["grass", "hydrant", "lab", "pillar", "road", "sky", "stair"]
    
    print("="*100)
    print("Scene별 자동 QP parameter 결정")
    print("="*100)
    
    results = []
    
    for scene in SCENES:
        csv_path = f'../comp_log/{scene}_qp37_trustmap.csv'
        
        try:
            # 자동 결정
            qp_param = determine_qp_parameter(csv_path, method='combined', debug=True)
            
            results.append({
                'Scene': scene,
                'Auto_QP': qp_param,
                'Manual_Optimal': {
                    'grass': 0.4, 'hydrant': 0.4, 'lab': 0.4, 'pillar': 0.4,
                    'road': 0.4, 'sky': 0.4, 'stair': 0.4
                }[scene]
            })
            
        except Exception as e:
            print(f"❌ {scene} 실패: {e}")
    
    # 결과 비교
    print("\n" + "="*100)
    print("자동 결정 vs 수동 최적값 비교")
    print("="*100)
    
    results_df = pd.DataFrame(results)
    results_df['Match'] = results_df['Auto_QP'] == results_df['Manual_Optimal']
    results_df['Diff'] = results_df['Auto_QP'] - results_df['Manual_Optimal']
    
    print("\n", results_df.to_string(index=False))
    
    accuracy = (results_df['Match'].sum() / len(results_df)) * 100
    print(f"\n정확도: {accuracy:.1f}% ({results_df['Match'].sum()}/{len(results_df)})")
    print(f"평균 오차: {abs(results_df['Diff']).mean():.2f}")