@echo off
setlocal enabledelayedexpansion
title CryptoAutoPilot - Admin Launcher
color 0A

:: Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PS_SCRIPT="%SCRIPT_DIR%start.ps1"

echo.
echo ==============================================================
echo              CryptoAutoPilot - Admin Launcher
echo ==============================================================
echo.

echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] PowerShell script: %PS_SCRIPT%
echo.

:: Check if the PowerShell script exists
if not exist %PS_SCRIPT% (
    echo [ERROR] PowerShell script not found: %PS_SCRIPT%
    echo [INFO] Please ensure start.ps1 is in the same directory as this batch file.
    goto :error
)

echo [INFO] Checking for administrator privileges...
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Requesting administrator privileges...
    echo [INFO] This will open a new administrator window.
    echo.
    
    :: Try to run PowerShell with admin rights
    powershell -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File %PS_SCRIPT% -Force' -Verb RunAs"
    
    echo [INFO] Admin request sent. Please approve the UAC prompt if it appears.
) else (
    echo [INFO] Administrator privileges detected.
    echo [INFO] Launching CryptoAutoPilot...
    echo.
    
    :: Run PowerShell script directly since we already have admin rights
    powershell -ExecutionPolicy Bypass -File %PS_SCRIPT% -Force
)

goto :end

:error
echo [ERROR] Launcher failed.
pause
exit /b 1

:end
echo.
echo [INFO] Launcher process completed.
if %errorlevel% neq 0 (
    echo [WARNING] Process exited with code: %errorlevel%
)
pause
exit /b 0
