# SatQuery AI — Unified Launcher
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "       Launching SatQuery AI System      " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$workspace = Get-Location

# Launch Backend in a new window
Write-Host "[1/2] Launching Backend on http://127.0.0.1:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$workspace'; .\backend\venv\Scripts\uvicorn.exe app.main:app --app-dir backend --reload --port 8000"

# Launch Frontend in a new window
Write-Host "[2/2] Launching Frontend on http://localhost:5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$workspace\frontend'; npm run dev"

Start-Sleep -Seconds 2
Write-Host ""
Write-Host "SatQuery AI is running!" -ForegroundColor Cyan
Write-Host "Dashboard:  http://localhost:5173" -ForegroundColor Yellow
Write-Host "API Docs:   http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
