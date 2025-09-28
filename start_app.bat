@echo off
title CryptoAutoPilot - Simple Launcher
color 0A

echo.
echo ==============================================================
echo              CryptoAutoPilot - Simple Launcher
echo ==============================================================
echo.

:: Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo [INFO] Please install Python 3.8 or higher from https://python.org
    echo [INFO] Then add Python to your PATH during installation.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python found.
echo.

:: Check if pip is available
echo [INFO] Checking pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not available!
    echo [INFO] Please ensure pip is installed with Python.
    echo.
    pause
    exit /b 1
)

echo [INFO] pip found.
echo.

:: Create virtual environment if it doesn't exist
echo [INFO] Setting up virtual environment...
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [INFO] Virtual environment activated.
echo.

:: Install required packages
echo [INFO] Installing required packages...
echo.

:: Core packages
echo [INFO] Installing core packages...
pip install numpy >nul 2>&1
echo [INFO] NumPy installed.

pip install requests >nul 2>&1
echo [INFO] Requests installed.

pip install scipy >nul 2>&1
echo [INFO] Scipy installed.

:: Quantum computing packages
echo [INFO] Installing quantum packages...
pip install qiskit >nul 2>&1
echo [INFO] Qiskit installed.

pip install qiskit-aer >nul 2>&1
echo [INFO] Qiskit Aer installed.

:: Machine learning packages
echo [INFO] Installing ML packages...
pip install torch >nul 2>&1
echo [INFO] PyTorch installed.

pip install tensorflow >nul 2>&1
echo [INFO] TensorFlow installed.

pip install scikit-learn >nul 2>&1
echo [INFO] Scikit-learn installed.

:: Web framework packages
echo [INFO] Installing web framework packages...
pip install flask >nul 2>&1
echo [INFO] Flask installed.

pip install flask-sqlalchemy >nul 2>&1
echo [INFO] Flask-SQLAlchemy installed.

pip install python-dotenv >nul 2>&1
echo [INFO] python-dotenv installed.

:: Other packages
echo [INFO] Installing additional packages...
pip install cryptography >nul 2>&1
echo [INFO] Cryptography installed.

pip install matplotlib >nul 2>&1
echo [INFO] Matplotlib installed.

pip install pandas >nul 2>&1
echo [INFO] Pandas installed.

pip install networkx >nul 2>&1
echo [INFO] NetworkX installed.

echo.
echo [INFO] All packages installed successfully!
echo.

:: Create necessary directories
echo [INFO] Creating directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "temp" mkdir temp
echo [INFO] Directories created.
echo.

:: Launch the application
echo [INFO] Starting CryptoAutoPilot Web Server...
echo [INFO] This will start a web server on http://localhost:5000
echo ==============================================================
echo.
python main.py

:: Keep window open
echo.
echo [INFO] Application finished.
echo [INFO] Press any key to close this window...
pause >nul
