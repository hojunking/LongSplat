"""
I-frame vs P-frame 분석 결과를 바탕으로 한 
Dynamic Grad Threshold 개선 방안

=== 분석 결과 핵심 발견 ===

1. I/P 비율 분포:
   - Grass: 1.66x (가장 낮음) → P-frame 비율 50%가 I-frame보다 큼
   - Lab: 1.69x
   - Road: 1.81x → P-frame 비율 25.6%가 I-frame보다 큼
   - Sky: 1.92x → P-frame 비율 22.9%가 I-frame보다 큼
   - Hydrant: 1.96x → P-frame 비율 49.1%가 I-frame보다 큼
   - Stair: 2.04x
   - Pillar: 2.15x (가장 높음) → P-frame 비율 1.5%만 I-frame보다 큼

2. P > I 비율과 QP parameter 관계:
   - Grass (P>I: 50.0%, P-ratio: 36.3%) → QP = 0.4 (적당)
   - Hydrant (P>I: 49.1%, P-ratio: 34.9%) → QP = 0.4 (적당)
   - Road (P>I: 25.6%, P-ratio: 37.7%) → QP = 0.5 (더 높음 필요)
   - Sky (P>I: 22.9%, P-ratio: 50.8%) → QP = 0.8 (가장 높음 필요)

3. 핵심 인사이트:
   ✅ I/P 비율이 낮을수록 bits 분포가 불균일
   ✅ P > I 비율이 높으면 예측 실패가 많음 → 변동성 큼
   ✅ P-frame 절대 비율도 중요 (Sky: 50.8%)
"""

import math
import torch
import numpy as np
import pandas as pd
import os


# =============================================================================
# 방법 1: I/P 비율 기반 QP parameter 조정 (추천!) ⭐
# =============================================================================

def calculate_ip_ratio_from_csv(csv_path):
    """
    CSV에서 I/P 비율 계산
    
    Returns:
        float: I-frame / P-frame 평균 bits 비율
    """
    if not os.path.exists(csv_path):
        return 2.0  # 기본값
    
    df = pd.read_csv(csv_path)
    df['Type'] = df['Type'].str.strip()
    
    i_mean = df[df['Type'] == 'I-SLICE']['Bits'].mean()
    p_mean = df[df['Type'] == 'P-SLICE']['Bits'].mean()
    
    if p_mean > 0:
        return i_mean / p_mean
    return 2.0


def get_qp_param_from_ip_ratio(ip_ratio):
    """
    I/P 비율에 따라 QP parameter 결정
    
    분석 결과:
    - I/P 비율 낮음 (< 1.7x) → bits 분포 불균일 → 낮은 QP로 충분
    - I/P 비율 보통 (1.7-2.0x) → 중간 QP
    - I/P 비율 높음 (≥ 2.0x) → bits 분포 균일 → 낮은 QP로 충분
    
    ⚠️ 하지만 실제로는 P-frame 비율이 더 중요!
    """
    if ip_ratio < 1.7:
        # Grass (1.66x), Lab (1.69x) - 하지만 둘 다 QP=0.4
        return 0.4
    elif ip_ratio < 1.85:
        # Road (1.81x) - QP=0.5 필요
        return 0.5
    elif ip_ratio < 2.0:
        # Sky (1.92x), Hydrant (1.96x) - Sky는 0.8, Hydrant는 0.4
        # 여기서 P-frame 비율을 봐야 함!
        return 0.6  # 중간값
    else:
        # Stair (2.04x), Pillar (2.15x) - 둘 다 QP=0.4
        return 0.4


# =============================================================================
# 방법 2: P > I 비율 기반 조정 (변동성 반영) 🔥
# =============================================================================

def calculate_p_larger_than_i_ratio(csv_path):
    """
    P-frame이 어떤 I-frame보다 큰 경우의 비율 계산
    
    Returns:
        float: P > I 비율 (0.0 ~ 1.0)
    """
    if not os.path.exists(csv_path):
        return 0.0
    
    df = pd.read_csv(csv_path)
    df['Type'] = df['Type'].str.strip()
    
    i_frames = df[df['Type'] == 'I-SLICE']
    p_frames = df[df['Type'] == 'P-SLICE']
    
    if len(i_frames) == 0 or len(p_frames) == 0:
        return 0.0
    
    i_min = i_frames['Bits'].min()
    
    # P-frame이 최소 I-frame보다 큰 경우
    p_larger_count = (p_frames['Bits'] > i_min).sum()
    
    return p_larger_count / len(p_frames)


def get_qp_adjustment_from_p_larger_ratio(p_larger_ratio):
    """
    P > I 비율에 따른 QP adjustment
    
    분석:
    - Grass (50.0%) / Hydrant (49.1%) → 변동성 큼 → QP 낮춤 (0.4)
    - Road (25.6%) / Sky (22.9%) → 변동성 보통 → QP 높임 (0.5, 0.8)
    - Stair (3.4%) / Pillar (1.5%) → 안정적 → QP 낮춤 (0.4)
    
    역설: P > I 비율이 높으면 오히려 낮은 QP 사용?
    → 아니다! P-frame 절대 비율과 함께 봐야 함!
    """
    # 단독으로는 명확한 패턴이 없음
    # P-frame 비율과 결합 필요
    return 0.0


# =============================================================================
# 방법 3: 종합 방식 (P-frame 비율 + I/P 비율 + P>I 비율) ⭐⭐⭐
# =============================================================================

def calculate_comprehensive_qp_param(csv_path, debug=False):
    """
    종합적 분석을 통한 QP parameter 결정
    
    고려 요소:
    1. P-frame 비율 (가장 중요)
    2. I/P bits 비율
    3. P > I 변동성
    """
    if not os.path.exists(csv_path):
        return 0.4
    
    df = pd.read_csv(csv_path)
    df['Type'] = df['Type'].str.strip()
    
    total_frames = len(df)
    
    # 1. P-frame 비율
    p_count = (df['Type'] == 'P-SLICE').sum()
    p_ratio = p_count / total_frames
    
    # 2. I/P bits 비율
    i_frames = df[df['Type'] == 'I-SLICE']
    p_frames = df[df['Type'] == 'P-SLICE']
    
    if len(i_frames) == 0 or len(p_frames) == 0:
        return 0.4
    
    i_mean = i_frames['Bits'].mean()
    p_mean = p_frames['Bits'].mean()
    ip_ratio = i_mean / p_mean if p_mean > 0 else 2.0
    
    # 3. P > I 비율
    i_min = i_frames['Bits'].min()
    p_larger_count = (p_frames['Bits'] > i_min).sum()
    p_larger_ratio = p_larger_count / len(p_frames)
    
    # 4. bits 변동성 (CV)
    bits = df['Bits'].values
    cv = (np.std(bits) / np.mean(bits)) * 100
    
    # === QP Parameter 결정 로직 ===
    
    # 기본값
    qp_param = 0.4
    
    # Rule 1: P-frame 비율이 압도적으로 높으면 QP 증가 (Sky 케이스)
    if p_ratio > 0.5:
        qp_param = 0.7
        if debug:
            print(f"  [Rule 1] P-frame 비율 높음 ({p_ratio:.1%}) → QP = {qp_param}")
    
    # Rule 2: P-frame 비율 40% 이상이고 I/P 비율 낮으면 QP 증가
    elif p_ratio > 0.4 and ip_ratio < 1.85:
        qp_param = 0.5
        if debug:
            print(f"  [Rule 2] P-ratio={p_ratio:.1%}, I/P={ip_ratio:.2f} → QP = {qp_param}")
    
    # Rule 3: I/P 비율이 매우 낮고 P>I 비율 높으면 변동성 큼
    elif ip_ratio < 1.7 and p_larger_ratio > 0.4:
        qp_param = 0.4
        if debug:
            print(f"  [Rule 3] 낮은 I/P ({ip_ratio:.2f}), 높은 P>I ({p_larger_ratio:.1%}) → QP = {qp_param}")
    
    # Rule 4: 안정적인 경우 (I/P ≥ 2.0, P>I < 10%)
    elif ip_ratio >= 2.0 and p_larger_ratio < 0.1:
        qp_param = 0.4
        if debug:
            print(f"  [Rule 4] 안정적 (I/P={ip_ratio:.2f}, P>I={p_larger_ratio:.1%}) → QP = {qp_param}")
    
    # Rule 5: 기타 중간 케이스
    else:
        # CV 기반 미세 조정
        if cv > 55:
            qp_param = 0.5
        else:
            qp_param = 0.4
        if debug:
            print(f"  [Rule 5] 기본 (CV={cv:.1f}%) → QP = {qp_param}")
    
    if debug:
        print(f"\n  최종 QP parameter: {qp_param}")
        print(f"  특성: P-ratio={p_ratio:.1%}, I/P={ip_ratio:.2f}, P>I={p_larger_ratio:.1%}, CV={cv:.1f}%")
    
    return qp_param


# =============================================================================
# 방법 4: Frame-type aware dynamic threshold (프레임 타입 고려) 🔥🔥
# =============================================================================

def get_dynamic_grad_threshold_v2(
    grad_threshold,
    bit_trust,
    frame_trust,
    current_frame_type,  # 새로 추가!
    qp_param_base=0.4,
    ip_ratio=2.0,
    debug=False
):
    """
    프레임 타입을 고려한 동적 threshold 계산
    
    핵심 아이디어:
    - I-frame: bit_trust 높음 → 더 보수적으로 (threshold 높임)
    - P-frame: 상황에 따라 다름
      - I/P 비율 낮으면 → P-frame도 클 수 있음 → 조심
      - I/P 비율 높으면 → P-frame 작음 → 공격적으로
    
    Args:
        grad_threshold: 기본 threshold
        bit_trust: 현재 프레임의 bit trust (0.0 ~ 0.5)
        frame_trust: 프레임 신뢰도
        current_frame_type: 'I-SLICE', 'P-SLICE', 'B-SLICE', 'b-SLICE'
        qp_param_base: 기본 QP parameter
        ip_ratio: I-frame / P-frame bits 비율
    """
    
    # 기존 수식
    # dynamic_grad_threshold = grad_threshold * exp(qp_param - (bit_trust + frame_trust))
    
    # 개선된 수식: 프레임 타입별 조정
    
    if current_frame_type == 'I-SLICE':
        # I-frame: 항상 bit_trust가 높음 (0.4~0.5)
        # → 과도한 growing 방지하기 위해 QP 증가
        qp_effective = qp_param_base + 0.2
        
        if debug:
            print(f"  [I-frame] QP 증가: {qp_param_base} → {qp_effective}")
    
    elif current_frame_type == 'P-SLICE':
        # P-frame: I/P 비율에 따라 조정
        
        if ip_ratio < 1.7:
            # I/P 비율 낮음 → P-frame도 클 수 있음
            # → bit_trust 높을 때 조심
            qp_effective = qp_param_base + 0.1
            if debug:
                print(f"  [P-frame, low I/P] QP 약간 증가: {qp_param_base} → {qp_effective}")
        
        elif ip_ratio > 2.0:
            # I/P 비율 높음 → P-frame 안정적으로 작음
            # → 기본 QP 사용
            qp_effective = qp_param_base
            if debug:
                print(f"  [P-frame, high I/P] QP 유지: {qp_effective}")
        
        else:
            # 중간
            qp_effective = qp_param_base
    
    else:
        # B-frame, b-frame: 보통 작음 → 기본 QP
        qp_effective = qp_param_base - 0.1
        qp_effective = max(qp_effective, 0.3)  # 최소값
    
    # 최종 threshold 계산
    dynamic_grad_threshold = grad_threshold * math.exp(
        qp_effective - (bit_trust + frame_trust)
    )
    
    if debug:
        print(f"  bit_trust={bit_trust:.3f}, frame_trust={frame_trust:.3f}")
        print(f"  qp_effective={qp_effective:.2f}")
        print(f"  threshold: {grad_threshold:.6f} → {dynamic_grad_threshold:.6f}")
    
    return dynamic_grad_threshold


# =============================================================================
# 통합 버전: adjust_anchor_heejung_song 수정
# =============================================================================

def adjust_anchor_heejung_song_improved(
    self,
    check_interval=100,
    success_threshold=0.8,
    grad_threshold=0.0002,
    min_opacity=0.005,
    require_purning=True,
    frame_trust=1.0,
    bit_trust=0.0,
    current_frame_type='P-SLICE',  # 🌟 새로 추가!
    qp_csv_path=None,              # 🌟 새로 추가!
    debug=False,
    mu=0.3,
):
    """
    I/P 분석 결과를 반영한 개선된 adjust_anchor
    """
    
    # =========================================================
    # 🔹 1. QP parameter 자동 결정 (scene-level)
    # =========================================================
    if qp_csv_path is not None and os.path.exists(qp_csv_path):
        # CSV에서 종합 분석
        qp_param_base = calculate_comprehensive_qp_param(qp_csv_path, debug=debug)
        
        # I/P 비율 계산
        ip_ratio = calculate_ip_ratio_from_csv(qp_csv_path)
    else:
        # 기본값
        qp_param_base = 0.4
        ip_ratio = 2.0
    
    # =========================================================
    # 🔹 2. 프레임 타입 기반 동적 threshold 계산
    # =========================================================
    dynamic_grad_threshold = get_dynamic_grad_threshold_v2(
        grad_threshold=grad_threshold,
        bit_trust=bit_trust,
        frame_trust=frame_trust,
        current_frame_type=current_frame_type,
        qp_param_base=qp_param_base,
        ip_ratio=ip_ratio,
        debug=debug
    )
    
    if debug:
        print(f"[Adjust Anchor] frame_type={current_frame_type}, "
              f"bit={bit_trust:.3f}, frame={frame_trust:.3f}")
        print(f"  → grad_th {grad_threshold:.5f} → {dynamic_grad_threshold:.5f}")
    
    # =========================================================
    # 기존 로직 (동일)
    # =========================================================
    grads = self.offset_gradient_accum / self.offset_denom
    grads[grads.isnan()] = 0.0
    grads_norm = torch.norm(grads, dim=-1)
    offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)
    
    self.anchor_growing(grads_norm, dynamic_grad_threshold, offset_mask)
    
    # ... (나머지 pruning 로직 동일) ...


# =============================================================================
# 사용 예시
# =============================================================================

"""
# 방법 1: Scene-level만 사용 (간단)
qp_param = calculate_comprehensive_qp_param('../comp_log/sky_qp37_trustmap.csv', debug=True)
# 출력: Sky → QP = 0.7 (P-frame 비율 50.8%)

dynamic_threshold = grad_threshold * math.exp(qp_param - (bit_trust + frame_trust))

# 방법 2: Frame-type aware (더 정교)
for iteration in training_loop:
    current_frame_id = get_current_frame_id()
    current_frame_type = get_frame_type(current_frame_id)  # 'I-SLICE', 'P-SLICE', ...
    
    gaussians.adjust_anchor_heejung_song_improved(
        frame_trust=frame_trust,
        bit_trust=bit_trust,
        current_frame_type=current_frame_type,
        qp_csv_path='../comp_log/scene_qp37_trustmap.csv',
        debug=(iteration % 1000 == 0)
    )
"""

# =============================================================================
# 실험 검증
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("I/P 비율 기반 QP Parameter 자동 결정 테스트")
    print("="*80)
    
    SCENES = ["grass", "hydrant", "lab", "pillar", "road", "sky", "stair"]
    GROUND_TRUTH = {
        "grass": 0.4,
        "hydrant": 0.4,
        "lab": 0.4,
        "pillar": 0.4,
        "road": 0.5,
        "sky": 0.8,
        "stair": 0.4
    }
    
    for scene in SCENES:
        csv_path = f'../comp_log/{scene}_qp37_trustmap.csv'
        
        if not os.path.exists(csv_path):
            continue
        
        print(f"\n{'='*80}")
        print(f"장면: {scene.upper()}")
        print('='*80)
        
        # 예측
        predicted_qp = calculate_comprehensive_qp_param(csv_path, debug=True)
        
        # 실제
        actual_qp = GROUND_TRUTH[scene]
        
        # 비교
        error = abs(predicted_qp - actual_qp)
        print(f"\n  예측: {predicted_qp:.1f}")
        print(f"  실제: {actual_qp:.1f}")
        print(f"  오차: {error:.1f}")
        
        if error == 0:
            print("  ✅ 정확!")
        elif error <= 0.1:
            print("  🟢 거의 정확")
        elif error <= 0.2:
            print("  🟡 약간 차이")
        else:
            print("  🔴 큰 차이")