@echo off
title XAUUSD Predictor Production Suite
set XAUUSD_API_KEY=gold-standard-2026
set PYTHONPATH=%~dp0

echo ===================================================
echo 🚀 XAUUSD Predictor Production Suite Launcher 🚀
echo ===================================================
echo.

echo [1/3] Starting FastAPI Backend on port 8000...
start "XAUUSD Predictor API" cmd /k "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo.
echo [2/3] Waiting 5 seconds for API models to load...
timeout /t 5 > nul

echo.
echo [3/3] Starting React Vite Frontend on port 5173...
start "XAUUSD Predictor Frontend" cmd /k "npm run dev --prefix frontend"

echo.
echo ===================================================
echo ✅ Production services launched successfully!
echo.
echo 🌐 API Endpoint:   http://127.0.0.1:8000
echo 🌐 Frontend URL:   http://localhost:5173
echo ===================================================
echo.
echo Note: Live server logs are running in the newly opened windows.
echo To stop the services, close the open CMD windows or run: taskkill /F /IM python.exe /T
echo.
pause
