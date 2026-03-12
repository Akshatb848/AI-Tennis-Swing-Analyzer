# TennisIQ — Production Startup Script
# Usage: .\start.ps1
# Starts both backend (port 8000) and frontend (port 3000) automatically.

Write-Host "=== TennisIQ Startup ===" -ForegroundColor Cyan

# ── 1. Kill anything on port 8000 ────────────────────────────────────────────
$existing = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique
foreach ($pid in $existing) {
    if ($pid -gt 4) {
        try { Stop-Process -Id $pid -Force; Write-Host "  Killed stale backend PID $pid" -ForegroundColor Yellow } catch {}
    }
}

# ── 2. Kill anything on port 3000 ────────────────────────────────────────────
$existing3000 = (Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique
foreach ($pid in $existing3000) {
    if ($pid -gt 4) {
        try { Stop-Process -Id $pid -Force; Write-Host "  Killed stale frontend PID $pid" -ForegroundColor Yellow } catch {}
    }
}
Start-Sleep -Seconds 1

# ── 3. Install frontend dependencies if needed ──────────────────────────────
$frontendDir = Join-Path $PSScriptRoot "tennis\frontend"
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "  Installing frontend dependencies..." -ForegroundColor Yellow
    Start-Process -FilePath "npm" -ArgumentList "install" -WorkingDirectory $frontendDir -NoNewWindow -Wait
}

# ── 4. Start backend ────────────────────────────────────────────────────────
Write-Host "  Starting backend on port 8000..." -ForegroundColor Green
$projectRoot = $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$projectRoot'; Write-Host 'Backend starting...' -ForegroundColor Green; python -m uvicorn tennis.api.app:app --port 8000 --reload"

Start-Sleep -Seconds 4

# ── 5. Verify backend ───────────────────────────────────────────────────────
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Host "  Backend healthy: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "  Backend did not respond in time — check the backend window for errors." -ForegroundColor Red
}

# ── 6. Start frontend ───────────────────────────────────────────────────────
Write-Host "  Starting frontend on port 3000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$frontendDir'; Write-Host 'Frontend starting...' -ForegroundColor Green; npx next dev"

Start-Sleep -Seconds 8

# ── 7. Verify frontend ──────────────────────────────────────────────────────
try {
    $fe = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -UseBasicParsing
    Write-Host "  Frontend running at http://localhost:3000 (status $($fe.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "  Frontend did not respond in time — check the frontend window for errors." -ForegroundColor Red
}

Write-Host ""
Write-Host "  Open: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan
