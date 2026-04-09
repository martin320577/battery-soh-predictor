"""Streamlit 기반 폐배터리 SOH 진단 대시보드"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from src.features import extract_single_cycle_features
from src.model import load_model
from src.predict import get_battery_grade

# 페이지 설정
st.set_page_config(
    page_title="폐배터리 SOH 진단 AI",
    page_icon="🔋",
    layout="wide",
)

# 커스텀 CSS
st.markdown("""
<style>
    .grade-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔋 폐배터리 SOH 빠른 진단 AI")
st.markdown("짧은 충방전 측정 데이터만으로 배터리 건강 상태(SOH)를 예측합니다.")

# 모델 로드
@st.cache_resource
def get_model():
    try:
        model, scaler, meta = load_model('models')
        return model, scaler, meta
    except Exception:
        return None, None, None

model, scaler, meta = get_model()

if model is None:
    st.error("⚠️ 학습된 모델이 없습니다. 먼저 `python run_pipeline.py`를 실행하세요.")
    st.stop()

# 사이드바
st.sidebar.header("📊 진단 설정")
st.sidebar.markdown("---")

# 등급 기준 안내
st.sidebar.markdown("### 배터리 등급 기준")
grade_info = {
    'A (≥80%)': ('재사용 가능', '#2ecc71'),
    'B (60-80%)': ('제한적 재사용', '#f39c12'),
    'C (40-60%)': ('재활용 권장', '#e67e22'),
    'D (<40%)': ('폐기 권장', '#e74c3c'),
}
for grade, (desc, color) in grade_info.items():
    st.sidebar.markdown(
        f'<span style="color:{color}; font-weight:bold">● {grade}</span> — {desc}',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 모델 정보")
st.sidebar.text(f"모델: {meta.get('best_model', 'N/A')}")
if 'xgb_metrics' in meta:
    m = meta['xgb_metrics'] if meta['best_model'] == 'XGBoost' else meta['rf_metrics']
    st.sidebar.text(f"MAE: {m.get('MAE', 'N/A')}%")
    st.sidebar.text(f"R²: {m.get('R2', 'N/A')}")

# 메인 콘텐츠
tab1, tab2, tab3 = st.tabs(["📤 SOH 진단", "📈 분석 리포트", "ℹ️ 사용 가이드"])

with tab1:
    st.header("배터리 데이터 업로드")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "충방전 측정 데이터 (CSV)",
            type=['csv'],
            help="필수 컬럼: Time, Voltage_measured. 선택: Current_measured, Temperature_measured"
        )

        # 샘플 데이터 사용 옵션
        use_sample = st.checkbox("샘플 데이터로 테스트")

        if use_sample and os.path.exists('data/sample_input.csv'):
            uploaded_file = 'data/sample_input.csv'

    if uploaded_file is not None:
        try:
            if isinstance(uploaded_file, str):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # 컬럼명 매핑
            col_map = {}
            for target, alts in {
                'Voltage_measured': ['voltage', 'Voltage', 'V', 'voltage_measured'],
                'Time': ['time', 't', 'Time_s', 'time_s'],
                'Current_measured': ['current', 'Current', 'I', 'current_measured'],
                'Temperature_measured': ['temperature', 'Temperature', 'T', 'temp', 'temperature_measured'],
            }.items():
                if target not in df.columns:
                    for alt in alts:
                        if alt in df.columns:
                            col_map[alt] = target
                            break
            if col_map:
                df = df.rename(columns=col_map)

            with col_upload:
                st.markdown("#### 업로드된 데이터")
                st.dataframe(df.head(10), use_container_width=True)

                # 전압 곡선 시각화
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df['Time'], df['Voltage_measured'], color='steelblue', linewidth=1.5)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Voltage (V)')
                ax.set_title('Uploaded Voltage Curve')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # 특성 추출 및 예측
            cycle_data = {
                'Voltage_measured': df['Voltage_measured'].values,
                'Time': df['Time'].values,
            }
            for col in ['Current_measured', 'Temperature_measured']:
                if col in df.columns:
                    cycle_data[col] = df[col].values

            features = extract_single_cycle_features(cycle_data)
            feature_cols = meta['feature_cols']
            X = np.array([[features.get(col, 0) for col in feature_cols]])
            X_scaled = scaler.transform(X)
            soh = float(model.predict(X_scaled)[0])
            soh = np.clip(soh, 0, 100)

            grade, desc, color = get_battery_grade(soh)

            with col_result:
                st.markdown("#### 진단 결과")

                # SOH 수치
                st.metric("SOH (State of Health)", f"{soh:.1f}%")

                # 등급 표시
                st.markdown(
                    f'<div class="grade-box" style="background-color:{color}">'
                    f'등급: {grade}<br><small>{desc}</small></div>',
                    unsafe_allow_html=True,
                )

                # 게이지 시각화
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh([0], [soh], color=color, height=0.5)
                ax.barh([0], [100], color='#ecf0f1', height=0.5, zorder=0)
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xlabel('SOH (%)')
                ax.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='EOL (80%)')
                ax.legend()
                ax.set_title('Battery Health')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # 판정 메시지
                if grade == 'A':
                    st.success("✅ 이 배터리는 ESS 등 2차 활용이 가능합니다.")
                elif grade == 'B':
                    st.warning("⚠️ 제한적 조건에서 재사용 가능합니다.")
                elif grade == 'C':
                    st.warning("🔄 소재 회수를 위한 재활용을 권장합니다.")
                else:
                    st.error("❌ 안전한 폐기 처리가 필요합니다.")

                # 주요 특성값 표시
                st.markdown("#### 추출된 주요 특성")
                feat_display = {
                    'voltage_slope': '전압 기울기',
                    'time_3_5_to_4_0': '3.5→4.0V 도달시간',
                    'temp_max': '최고 온도 (°C)',
                    'internal_resistance': '내부저항 추정',
                    'voltage_area': '전압 곡선 면적',
                    'total_time': '총 측정 시간',
                }
                feat_rows = []
                for k, label in feat_display.items():
                    if k in features and features[k] is not None:
                        feat_rows.append({'특성': label, '값': f"{features[k]:.4f}"})
                st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"오류 발생: {e}")

with tab2:
    st.header("📈 모델 분석 리포트")

    # 저장된 그래프 표시
    report_images = {
        '용량 감소 추이': 'outputs/capacity_degradation.png',
        '충방전 곡선': 'outputs/voltage_curves.png',
        '예측 성능 비교': 'outputs/prediction_comparison.png',
        '특성 중요도': 'outputs/feature_importance.png',
        '특성 상관관계': 'outputs/correlation_heatmap.png',
    }

    for title, path in report_images.items():
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)
        else:
            st.info(f"'{path}' 파일을 찾을 수 없습니다.")

    # 모델 성능 비교 테이블
    if 'rf_metrics' in meta and 'xgb_metrics' in meta:
        st.subheader("모델 성능 비교")
        perf_df = pd.DataFrame({
            '지표': ['MAE (%)', 'RMSE (%)', 'R²'],
            'Random Forest': [
                meta['rf_metrics'].get('MAE', '-'),
                meta['rf_metrics'].get('RMSE', '-'),
                meta['rf_metrics'].get('R2', '-'),
            ],
            'XGBoost': [
                meta['xgb_metrics'].get('MAE', '-'),
                meta['xgb_metrics'].get('RMSE', '-'),
                meta['xgb_metrics'].get('R2', '-'),
            ],
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("ℹ️ 사용 가이드")
    st.markdown("""
    ### 데이터 형식
    CSV 파일로 충방전 측정 데이터를 업로드하세요.

    **필수 컬럼:**
    | 컬럼명 | 설명 | 단위 |
    |--------|------|------|
    | `Time` | 측정 시간 | 분 (min) |
    | `Voltage_measured` | 측정 전압 | V |

    **선택 컬럼 (정확도 향상):**
    | 컬럼명 | 설명 | 단위 |
    |--------|------|------|
    | `Current_measured` | 측정 전류 | A |
    | `Temperature_measured` | 측정 온도 | °C |

    ### 충전 데이터 권장 시간
    | 구분 | 시간 | 설명 |
    |------|------|------|
    | **최소** | 5~10분 | 예측 가능하지만 정확도 낮음 |
    | **권장** | 20~30분 | 높은 정확도로 진단 가능 |
    | 기존 방식 | 2~4시간 | 완전 충방전 필요 (본 AI 불필요) |

    > 💡 본 AI를 사용하면 기존 대비 **진단 시간 90% 이상 단축** 가능

    ### 진단 과정
    1. CSV 파일 업로드 (또는 샘플 데이터 사용)
    2. AI가 전압 곡선에서 특성을 자동 추출
    3. 학습된 모델로 SOH 예측
    4. 등급 판정 및 활용 방안 제시

    ### 배터리 등급
    - **A등급 (SOH ≥ 80%)**: ESS, 저속 전기차 등 2차 활용 가능
    - **B등급 (60~80%)**: 제한적 조건에서 재사용 가능
    - **C등급 (40~60%)**: 소재 회수(리사이클링) 권장
    - **D등급 (< 40%)**: 안전 폐기 처리 필요
    """)

    st.markdown("---")
    st.markdown("**개발**: 폐배터리 SOH 진단 AI 프로젝트")
    st.markdown("**데이터**: NASA PCoE 리튬이온 배터리 데이터셋 기반")
