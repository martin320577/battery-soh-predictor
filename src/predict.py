"""SOH 예측 실행 모듈"""
import numpy as np
import pandas as pd
from src.features import extract_single_cycle_features, get_feature_columns
from src.model import load_model


def predict_soh(cycle_data, model=None, scaler=None, meta=None, model_dir='models'):
    """단일 충방전 사이클 데이터에서 SOH 예측"""
    if model is None:
        model, scaler, meta = load_model(model_dir)

    feature_cols = meta['feature_cols']
    features = extract_single_cycle_features(cycle_data)

    if features is None:
        return None

    X = np.array([[features.get(col, 0) for col in feature_cols]])
    X_scaled = scaler.transform(X)
    soh = model.predict(X_scaled)[0]
    soh = np.clip(soh, 0, 100)

    return soh


def get_battery_grade(soh):
    """SOH에 따른 배터리 등급 판정"""
    if soh >= 80:
        return 'A', '재사용 가능', '#2ecc71'
    elif soh >= 60:
        return 'B', '제한적 재사용', '#f39c12'
    elif soh >= 40:
        return 'C', '재활용 권장', '#e67e22'
    else:
        return 'D', '폐기 권장', '#e74c3c'


def predict_from_csv(csv_path, model_dir='models'):
    """CSV 파일에서 배터리 데이터를 읽어 SOH 예측"""
    model, scaler, meta = load_model(model_dir)
    df = pd.read_csv(csv_path)

    required = ['Voltage_measured', 'Time']
    for col in required:
        if col not in df.columns:
            # 유사한 컬럼명 시도
            alternatives = {
                'Voltage_measured': ['voltage', 'Voltage', 'V'],
                'Time': ['time', 't', 'Time_s'],
            }
            found = False
            for alt in alternatives.get(col, []):
                if alt in df.columns:
                    df = df.rename(columns={alt: col})
                    found = True
                    break
            if not found:
                raise ValueError(f"필수 컬럼 '{col}' 없음. 컬럼: {list(df.columns)}")

    cycle_data = {
        'Voltage_measured': df['Voltage_measured'].values,
        'Time': df['Time'].values,
    }
    for col in ['Current_measured', 'Temperature_measured']:
        if col in df.columns:
            cycle_data[col] = df[col].values

    soh = predict_soh(cycle_data, model, scaler, meta)
    grade, desc, color = get_battery_grade(soh)

    return {
        'soh': round(soh, 2),
        'grade': grade,
        'grade_desc': desc,
        'grade_color': color,
    }
