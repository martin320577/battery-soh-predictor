"""SOH 예측 모델 학습 및 평가 모듈"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import json


def prepare_data(df, feature_cols, test_size=0.2, random_state=42):
    """학습/테스트 데이터 분할"""
    X = df[feature_cols].values
    y = df['soh'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_random_forest(X_train, y_train, **kwargs):
    """Random Forest 모델 학습"""
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
    }
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """XGBoost 모델 학습"""
    params = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }
    params.update(kwargs)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(model, X_test, y_test):
    """모델 성능 평가"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'y_test': y_test,
        'y_pred': y_pred,
    }


def get_feature_importance(model, feature_cols):
    """특성 중요도 반환"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_cols))

    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    return imp_df


def train_and_compare(df, feature_cols):
    """Random Forest와 XGBoost를 학습하고 비교"""
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, feature_cols)

    print("=" * 50)
    print("Random Forest 학습 중...")
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    print(f"  MAE: {rf_results['MAE']:.4f}%")
    print(f"  RMSE: {rf_results['RMSE']:.4f}%")
    print(f"  R²: {rf_results['R2']:.4f}")

    print("\nXGBoost 학습 중...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test)
    print(f"  MAE: {xgb_results['MAE']:.4f}%")
    print(f"  RMSE: {xgb_results['RMSE']:.4f}%")
    print(f"  R²: {xgb_results['R2']:.4f}")

    # 더 나은 모델 선택
    if xgb_results['MAE'] <= rf_results['MAE']:
        best_model = xgb_model
        best_name = 'XGBoost'
        best_results = xgb_results
    else:
        best_model = rf_model
        best_name = 'Random Forest'
        best_results = rf_results

    print(f"\n최적 모델: {best_name} (MAE: {best_results['MAE']:.4f}%)")
    print("=" * 50)

    return {
        'rf': {'model': rf_model, 'results': rf_results},
        'xgb': {'model': xgb_model, 'results': xgb_results},
        'best': {'model': best_model, 'name': best_name, 'results': best_results},
        'scaler': scaler,
        'feature_cols': feature_cols,
    }


def save_model(model_data, save_dir='models'):
    """학습된 모델과 스케일러 저장"""
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model_data['best']['model'],
                os.path.join(save_dir, 'best_model.pkl'))
    joblib.dump(model_data['scaler'],
                os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(model_data['rf']['model'],
                os.path.join(save_dir, 'rf_model.pkl'))
    joblib.dump(model_data['xgb']['model'],
                os.path.join(save_dir, 'xgb_model.pkl'))

    # 메타데이터 저장
    meta = {
        'best_model': model_data['best']['name'],
        'feature_cols': model_data['feature_cols'],
        'rf_metrics': {k: v for k, v in model_data['rf']['results'].items()
                       if k not in ('y_test', 'y_pred')},
        'xgb_metrics': {k: v for k, v in model_data['xgb']['results'].items()
                        if k not in ('y_test', 'y_pred')},
    }
    with open(os.path.join(save_dir, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"모델 저장 완료: {save_dir}/")


def load_model(save_dir='models'):
    """저장된 모델과 스케일러 로드"""
    model = joblib.load(os.path.join(save_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
    with open(os.path.join(save_dir, 'model_meta.json'), 'r') as f:
        meta = json.load(f)
    return model, scaler, meta
