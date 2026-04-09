@echo off
echo ========================================
echo  배터리 SOH 진단 AI - 자동 설치 및 실행
echo ========================================
echo.

echo [1/3] 가상환경 생성 중...
python -m venv venv
call venv\Scripts\activate

echo [2/3] 라이브러리 설치 중...
pip install -r requirements.txt

echo [3/3] 대시보드 실행 중...
echo.
echo 브라우저에서 http://localhost:8501 접속하세요
echo.
streamlit run app.py
