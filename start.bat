@echo off
title DeepFaceLive - Startup

echo ========================================
echo   DeepFaceLive Startup Script
echo ========================================
echo.

:: Start Backend (in new window)
echo Starting Backend Server...
start "Backend - FastAPI" cmd /k "cd /d %~dp0backend && call conda activate face_recog && python server.py"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak > nul

:: Start Frontend (in new window)
echo Starting Frontend Dev Server...
start "Frontend - React" cmd /k "cd /d %~dp0frontend && yarn start"

echo.
echo ========================================
echo   Both servers are starting...
echo   - Backend: http://localhost:8000
echo   - Frontend: http://localhost:3000
echo ========================================
echo.
echo You can close this window.
pause
