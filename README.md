# 🔋 폐배터리 SOH 빠른 진단 AI

짧은 충방전 측정 데이터(수 분)만으로 배터리 건강 상태(SOH)를 정확히 예측하는 AI 모델 및 웹 대시보드

## 배경

전기차 보급 확대로 폐배터리가 급증하고 있습니다. 배터리를 재사용(ESS)할지, 소재를 회수(재활용)할지 판단하려면 SOH를 알아야 하지만, 기존 방식은 완전 충방전에 수 시간이 소요됩니다. 본 프로젝트는 AI를 활용하여 **짧은 측정만으로 SOH를 추정**함으로써 진단 시간을 대폭 단축합니다.

## 모델 성능

| 모델 | MAE (%) | RMSE (%) | R² |
|------|---------|----------|-----|
| Random Forest | 1.59 | 2.31 | 0.9525 |
| XGBoost | 1.62 | 2.32 | 0.9523 |

> 목표 오차 5% 이내 달성 ✅

## 프로젝트 구조

```
battery-soh-predictor/
├── data/               # 배터리 데이터셋
├── models/             # 학습된 모델 파일
├── notebooks/          # 탐색용 노트북
├── outputs/            # 시각화 결과물
├── src/
│   ├── preprocess.py   # 데이터 전처리
│   ├── features.py     # 특성 추출
│   ├── model.py        # 모델 학습/평가
│   └── predict.py      # 예측 실행
├── app.py              # Streamlit 대시보드
├── run_pipeline.py     # 전체 파이프라인 실행
└── requirements.txt    # 라이브러리 목록
```

## 설치 및 실행

### 1. 환경 설정
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python run_pipeline.py
```

### 3. 대시보드 실행
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501` 접속

## 데이터

- **데이터셋**: NASA PCoE 리튬이온 배터리 데이터셋
- `.mat` 파일을 `data/` 폴더에 넣으면 자동 로딩
- 데이터 없이도 합성 데이터로 동작 가능

## 배터리 등급 기준

| 등급 | SOH | 판정 |
|------|-----|------|
| A | ≥ 80% | 재사용 가능 (ESS 등) |
| B | 60~80% | 제한적 재사용 |
| C | 40~60% | 재활용 권장 |
| D | < 40% | 폐기 권장 |

## 기술 스택

Python, NumPy, Pandas, scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## 입력 데이터 형식

CSV 파일, 필수 컬럼:
- `Time` — 측정 시간 (분)
- `Voltage_measured` — 측정 전압 (V)

선택 컬럼 (정확도 향상):
- `Current_measured` — 전류 (A)
- `Temperature_measured` — 온도 (°C)
