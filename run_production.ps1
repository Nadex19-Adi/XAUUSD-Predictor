# Production Launch Script for XAUUSD Predictor
# Starts API and Dashboard in background

$env:XAUUSD_API_KEY = "gold-standard-2026"
$env:PYTHONPATH = $PSScriptRoot

Write-Host "🚀 Starting XAUUSD Predictor Production Suite..." -ForegroundColor Cyan

# 1. Start FastAPI Backend
Write-Host "Starting API on port 8000..." -ForegroundColor Yellow
Start-Process python -ArgumentList "-m uvicorn api.main:app --host 0.0.0.0 --port 8000" -WindowStyle Hidden -PassThru

# 2. Wait for API to warm up
Write-Host "Waiting 10s for models to load..."
Start-Sleep -Seconds 10

# 3. Start Streamlit Dashboard
Write-Host "Starting Dashboard on port 8501..." -ForegroundColor Green
Start-Process streamlit -ArgumentList "run dashboard/app.py --server.port 8501" -WindowStyle Hidden -PassThru

Write-Host "`n✅ System is now running in the background!" -ForegroundColor White
Write-Host "API: http://localhost:8000"
Write-Host "Dashboard: http://localhost:8501"
Write-Host "`nTo stop everything, run: taskkill /F /IM python.exe /T" -ForegroundColor Gray
