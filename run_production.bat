@echo off
title XAUUSD Predictor Production Suite
set XAUUSD_API_KEY=gold-standard-2026
set PYTHONPATH=%~dp0

echo ===================================================
echo 🚀 XAUUSD Predictor Production Suite Launcher 🚀
echo ===================================================
echo.

echo [1/3] Starting FastAPI Backend on port 8000...
start "XAUUSD Predictor API" cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo.
echo [2/3] Waiting 5 seconds for API models to load...
timeout /t 5 > nul

echo.
echo [3/3] Starting Streamlit Dashboard on port 8501...
start "XAUUSD Predictor Dashboard" cmd /k "python -m streamlit run dashboard/app.py --server.port 8501"

echo.
echo ===================================================
echo ✅ Production services launched successfully!
echo.
echo 🌐 API Endpoint:   http://localhost:8000
echo 🌐 Dashboard URL:  http://localhost:8501
echo ===================================================
echo.
echo Note: Live server logs are running in the newly opened windows.
echo To stop the services, simply close the open CMD windows.
echo.
pause
