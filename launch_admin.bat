@echo off
setlocal enabledelayedexpansion
title CryptoAutoPilot - Advanced Admin Launcher
color 0A

:: Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set PS_SCRIPT="%SCRIPT_DIR%start.ps1"

echo.
echo ==============================================================
echo           CryptoAutoPilot - Advanced Admin Launcher
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

:: Check if PowerShell is available
powershell -Command "Exit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is not available or not working properly.
    echo [INFO] Please ensure PowerShell is installed and accessible.
    goto :error
)

echo [INFO] PowerShell is available.
echo.

:: Check for administrator privileges
echo [INFO] Checking for administrator privileges...
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Administrator privileges not detected.
    echo [INFO] Attempting to elevate privileges...
    echo.
    
    :: Method 1: Try using runas (alternative method)
    echo [INFO] Method 1: Using runas command...
    echo [INFO] You may need to enter administrator password.
    
    :: Get current username
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set datetime=%%a
    set temp_file=%temp%\admin_launcher_%datetime%.ps1
    
    :: Create a temporary PowerShell script
    echo Set-ExecutionPolicy Bypass -Scope Process -Force > %temp_file%
    echo $scriptPath = %PS_SCRIPT% >> %temp_file%
    echo if (Test-Path $scriptPath) { >> %temp_file%
    echo     try { >> %temp_file%
    echo         & $scriptPath -Force >> %temp_file%
    echo     } catch { >> %temp_file%
    echo         Write-Host "Error running script: $_" -ForegroundColor Red >> %temp_file%
    echo     } >> %temp_file%
    echo } else { >> %temp_file%
    echo     Write-Host "Script not found: $scriptPath" -ForegroundColor Red >> %temp_file%
    echo } >> %temp_file%
    echo Read-Host "Press Enter to exit" >> %temp_file%
    
    :: Try to run as administrator
    runas /user:Administrator "powershell -ExecutionPolicy Bypass -File %temp_file%"
    
    if %errorlevel% neq 0 (
        echo [WARNING] runas method failed.
        echo.
        
        :: Method 2: Try PowerShell elevation
        echo [INFO] Method 2: Using PowerShell elevation...
        echo [INFO] Please approve the UAC prompt if it appears.
        echo.
        
        powershell -Command ^
            "$script = %PS_SCRIPT%; " ^
            "if (Test-Path $script) { " ^
            "    Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File', $script, '-Force' -Verb RunAs -Wait; " ^
            "} else { " ^
            "    Write-Host 'Script not found: $script'; " ^
            "    Read-Host 'Press Enter to exit'; " ^
            "}"
        
        if %errorlevel% neq 0 (
            echo [WARNING] PowerShell elevation method failed.
            echo.
            
            :: Method 3: Manual instructions
            echo [INFO] Method 3: Manual elevation required.
            echo [INFO] Please follow these steps:
            echo.
            echo 1. Right-click on Command Prompt or PowerShell
            echo 2. Select "Run as administrator"
            echo 3. Navigate to: %SCRIPT_DIR%
            echo 4. Run: powershell -ExecutionPolicy Bypass -File start.ps1 -Force
            echo.
            echo [INFO] Alternatively, try running this batch file as administrator.
            echo.
        ) else (
            echo [INFO] PowerShell elevation method succeeded.
        )
    ) else (
        echo [INFO] runas method succeeded.
    )
    
    :: Clean up temp file
    if exist %temp_file% del %temp_file%
    
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
echo [INFO] Please check the following:
echo 1. PowerShell is installed and working
echo 2. start.ps1 exists in the same directory
echo 3. You have sufficient permissions
echo.
pause
exit /b 1

:end
echo.
echo [INFO] Launcher process completed.
if %errorlevel% neq 0 (
    echo [WARNING] Process exited with code: %errorlevel%
    echo [INFO] This may indicate that some dependencies failed to install.
    echo [INFO] Check the output above for specific error messages.
) else (
    echo [INFO] Process completed successfully.
)
echo.
pause
exit /b 0
