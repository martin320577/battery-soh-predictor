"""충방전 사이클에서 SOH 예측을 위한 특성 추출 모듈"""
import numpy as np
import pandas as pd


def extract_single_cycle_features(cycle_data):
    """단일 충방전 사이클에서 특성 추출"""
    V = cycle_data.get('Voltage_measured')
    I = cycle_data.get('Current_measured')
    T = cycle_data.get('Temperature_measured')
    t = cycle_data.get('Time')

    if V is None or len(V) < 10:
        return None

    features = {}

    # 1. 전압 관련 특성
    features['voltage_mean'] = np.mean(V)
    features['voltage_std'] = np.std(V)
    features['voltage_max'] = np.max(V)
    features['voltage_min'] = np.min(V)
    features['voltage_range'] = np.max(V) - np.min(V)

    # 2. 특정 전압 구간 도달 시간 (3.5V → 4.0V)
    idx_35 = np.where(V >= 3.5)[0]
    idx_40 = np.where(V >= 4.0)[0]
    if len(idx_35) > 0 and len(idx_40) > 0:
        features['time_3_5_to_4_0'] = t[idx_40[0]] - t[idx_35[0]]
    else:
        features['time_3_5_to_4_0'] = np.nan

    # 3. 전압 곡선 기울기 (선형 회귀)
    if len(t) > 1:
        coeffs = np.polyfit(t[:len(V)], V, 1)
        features['voltage_slope'] = coeffs[0]
    else:
        features['voltage_slope'] = 0

    # 4. 전압 곡선 아래 면적 (에너지 관련)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    if len(t) >= len(V):
        features['voltage_area'] = _trapz(V, t[:len(V)])
    else:
        features['voltage_area'] = _trapz(V)

    # 5. 전류 관련 특성
    if I is not None and len(I) > 0:
        features['current_mean'] = np.mean(np.abs(I))
        features['current_std'] = np.std(I)
    else:
        features['current_mean'] = np.nan
        features['current_std'] = np.nan

    # 6. 온도 관련 특성
    if T is not None and len(T) > 0:
        features['temp_max'] = np.max(T)
        features['temp_mean'] = np.mean(T)
        features['temp_rise'] = np.max(T) - np.min(T)
        features['temp_std'] = np.std(T)
    else:
        features['temp_max'] = np.nan
        features['temp_mean'] = np.nan
        features['temp_rise'] = np.nan
        features['temp_std'] = np.nan

    # 7. 내부저항 추정 (V/I)
    if I is not None and len(I) > 0:
        abs_I = np.abs(I)
        valid = abs_I > 0.01
        if np.any(valid):
            features['internal_resistance'] = np.mean(V[valid] / abs_I[valid])
        else:
            features['internal_resistance'] = np.nan
    else:
        features['internal_resistance'] = np.nan

    # 8. 시간 관련 특성
    features['total_time'] = t[-1] - t[0]

    # 9. 전압 미분 특성 (dV/dt)
    if len(V) > 2:
        dv = np.diff(V)
        dt_diff = np.diff(t[:len(V)])
        dt_diff[dt_diff == 0] = 1e-6
        dvdt = dv / dt_diff
        features['dvdt_mean'] = np.mean(dvdt)
        features['dvdt_std'] = np.std(dvdt)
        features['dvdt_max'] = np.max(np.abs(dvdt))
    else:
        features['dvdt_mean'] = 0
        features['dvdt_std'] = 0
        features['dvdt_max'] = 0

    return features


def extract_all_features(all_cells, cycle_type='charge'):
    """모든 셀의 모든 사이클에서 특성 추출 → DataFrame 반환"""
    rows = []

    for cell_name, cell_data in all_cells.items():
        init_cap = cell_data['initial_capacity']
        cycles = cell_data[cycle_type]

        for cycle in cycles:
            feats = extract_single_cycle_features(cycle)
            if feats is None:
                continue

            cap = cycle.get('Capacity', None)
            if cap is None:
                continue

            soh = (cap / init_cap) * 100.0
            feats['cell'] = cell_name
            feats['cycle'] = cycle['cycle_number']
            feats['capacity'] = cap
            feats['soh'] = soh

            rows.append(feats)

    df = pd.DataFrame(rows)

    # 결측치 처리
    feature_cols = [c for c in df.columns if c not in ['cell', 'cycle', 'capacity', 'soh']]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def get_feature_columns(df):
    """특성 컬럼 목록 반환 (타겟/메타 컬럼 제외)"""
    exclude = ['cell', 'cycle', 'capacity', 'soh']
    return [c for c in df.columns if c not in exclude]
