# Production Launch Script for XAUUSD Predictor with React Frontend
# Starts both FastAPI and React Vite app

$env:XAUUSD_API_KEY = "gold-standard-2026"
$env:PYTHONPATH = $PSScriptRoot

Write-Host "🚀 Starting XAUUSD Predictor React Suite..." -ForegroundColor Cyan

# 1. Start FastAPI Backend
Write-Host "Starting FastAPI Backend on port 8000..." -ForegroundColor Yellow
$apiProcess = Start-Process python -ArgumentList "-m uvicorn api.main:app --host 0.0.0.0 --port 8000" -WindowStyle Hidden -PassThru

# 2. Wait for API to warm up
Write-Host "Waiting 10s for RAG sharded memory and model to load..."
Start-Sleep -Seconds 10

# 3. Start React Frontend
Write-Host "Starting React Vite Frontend..." -ForegroundColor Green
$npmProcess = Start-Process cmd -ArgumentList "/c npm run dev --prefix frontend" -WindowStyle Normal -PassThru

Write-Host "`n✅ Both services started successfully!" -ForegroundColor White
Write-Host "• FastAPI Backend: http://127.0.0.1:8000"
Write-Host "• React Frontend:  http://localhost:5173"
Write-Host "`nTo stop the suite, terminate the terminal window or run: taskkill /F /IM python.exe /T" -ForegroundColor Gray
