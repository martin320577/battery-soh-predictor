@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  배터리 SOH 진단 AI - 자동 설치 및 실행
echo ========================================
echo.

if not exist venv (
    echo [1/3] 가상환경 생성 중...
    python -m venv venv
) else (
    echo [1/3] 가상환경 이미 존재
)

call venv\Scripts\activate.bat

echo [2/3] 라이브러리 확인 중...
pip install -r requirements.txt -q

echo [3/3] 대시보드 실행 중...
echo.
echo 브라우저에서 http://localhost:8501 접속하세요
echo.
streamlit run app.py

pause
