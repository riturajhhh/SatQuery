# SatQuery AI — Start Backend Server
Write-Host "Starting SatQuery AI Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
& ".\backend\venv\Scripts\uvicorn.exe" app.main:app --app-dir backend --reload --port 8000
