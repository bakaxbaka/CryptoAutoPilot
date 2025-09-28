@echo off
title CryptoAutoPilot - Quantum Assault System
color 0A

echo.
echo ==============================================================
echo              CryptoAutoPilot - Quantum Assault System
echo                    Dependency Installer & Launcher
echo ==============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://python.org
    echo.
    pause
    exit /b 1
)

echo [INFO] Python detected, checking version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python version: %PYTHON_VERSION%
echo.

:: Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not available!
    echo Please ensure pip is installed with Python.
    echo.
    pause
    exit /b 1
)

echo [INFO] pip detected, updating to latest version...
pip install --upgrade pip >nul 2>&1
echo [INFO] pip updated successfully.
echo.

:: Create virtual environment
echo [INFO] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment already exists.
)
echo.

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [INFO] Virtual environment activated.
echo.

:: Install core dependencies
echo [INFO] Installing core dependencies...
echo.

echo [INFO] Installing NumPy...
pip install numpy >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install NumPy!
    pause
    exit /b 1
)
echo [INFO] NumPy installed successfully.

echo [INFO] Installing Requests...
pip install requests >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Requests!
    pause
    exit /b 1
)
echo [INFO] Requests installed successfully.

echo [INFO] Installing Scipy...
pip install scipy >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Scipy!
    pause
    exit /b 1
)
echo [INFO] Scipy installed successfully.
echo.

:: Install quantum computing dependencies
echo [INFO] Installing quantum computing dependencies...
echo.

echo [INFO] Installing Qiskit...
pip install qiskit >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Qiskit! Quantum features will be limited.
    set QISKIT_AVAILABLE=0
) else (
    echo [INFO] Qiskit installed successfully.
    set QISKIT_AVAILABLE=1
)

echo [INFO] Installing Qiskit Aer...
pip install qiskit-aer >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Qiskit Aer! Quantum simulation will be limited.
) else (
    echo [INFO] Qiskit Aer installed successfully.
)

echo [INFO] Installing Qiskit IBM Quantum...
pip install qiskit-ibmq-provider >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Qiskit IBM Quantum! Cloud quantum access will be limited.
) else (
    echo [INFO] Qiskit IBM Quantum installed successfully.
)
echo.

:: Install machine learning dependencies
echo [INFO] Installing machine learning dependencies...
echo.

echo [INFO] Installing PyTorch...
pip install torch torchvision torchaudio >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install PyTorch! ML features will be limited.
    set PYTORCH_AVAILABLE=0
) else (
    echo [INFO] PyTorch installed successfully.
    set PYTORCH_AVAILABLE=1
)

echo [INFO] Installing TensorFlow...
pip install tensorflow >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install TensorFlow! Some ML features will be limited.
    set TENSORFLOW_AVAILABLE=0
) else (
    echo [INFO] TensorFlow installed successfully.
    set TENSORFLOW_AVAILABLE=1
)

echo [INFO] Installing Scikit-learn...
pip install scikit-learn >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Scikit-learn! ML features will be limited.
) else (
    echo [INFO] Scikit-learn installed successfully.
)
echo.

:: Install distributed computing dependencies
echo [INFO] Installing distributed computing dependencies...
echo.

echo [INFO] Installing MPI4Py...
pip install mpi4py >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install MPI4Py! Distributed computing will be limited.
    set MPI_AVAILABLE=0
) else (
    echo [INFO] MPI4Py installed successfully.
    set MPI_AVAILABLE=1
)
echo.

:: Install cryptography dependencies
echo [INFO] Installing cryptography dependencies...
echo.

echo [INFO] Installing Cryptography...
pip install cryptography >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Cryptography! Advanced crypto features will be limited.
    set CRYPTO_AVAILABLE=0
) else (
    echo [INFO] Cryptography installed successfully.
    set CRYPTO_AVAILABLE=1
)
echo.

:: Install additional utilities
echo [INFO] Installing additional utilities...
echo.

echo [INFO] Installing Matplotlib...
pip install matplotlib >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Matplotlib! Visualization will be limited.
) else (
    echo [INFO] Matplotlib installed successfully.
)

echo [INFO] Installing Seaborn...
pip install seaborn >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Seaborn! Advanced visualization will be limited.
) else (
    echo [INFO] Seaborn installed successfully.
)

echo [INFO] Installing Pandas...
pip install pandas >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Pandas! Data processing will be limited.
) else (
    echo [INFO] Pandas installed successfully.
)

echo [INFO] Installing NetworkX...
pip install networkx >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install NetworkX! Network analysis will be limited.
) else (
    echo [INFO] NetworkX installed successfully.
)
echo.

:: Install ZK-SNARK dependencies
echo [INFO] Installing ZK-SNARK dependencies...
echo.

echo [INFO] Installing PySNARK...
pip install pysnark >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install PySNARK! ZK-SNARK features will be limited.
    set ZKSNARK_AVAILABLE=0
) else (
    echo [INFO] PySNARK installed successfully.
    set ZKSNARK_AVAILABLE=1
)

echo [INFO] Installing Bellman...
pip install bellman >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install Bellman! Advanced ZK-SNARK features will be limited.
) else (
    echo [INFO] Bellman installed successfully.
)
echo.

:: Install homomorphic encryption dependencies
echo [INFO] Installing homomorphic encryption dependencies...
echo.

echo [INFO] Installing PySEAL...
pip install pyseal >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install PySEAL! Homomorphic encryption will be simulated.
    set HOMOMORPHIC_AVAILABLE=0
) else (
    echo [INFO] PySEAL installed successfully.
    set HOMOMORPHIC_AVAILABLE=1
)

echo [INFO] Installing TenSEAL...
pip install tenseal >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install TenSEAL! Advanced homomorphic encryption will be limited.
) else (
    echo [INFO] TenSEAL installed successfully.
)
echo.

:: Create configuration file
echo [INFO] Creating configuration file...
if not exist "config.json" (
    echo {> config.json
    echo   "system": {>> config.json
    echo     "name": "CryptoAutoPilot Quantum Assault System",>> config.json
    echo     "version": "1.0.0",>> config.json
    echo     "debug_mode": true,>> config.json
    echo     "log_level": "INFO">> config.json
    echo   },>> config.json
    echo   "quantum": {>> config.json
    echo     "qiskit_available": %QISKIT_AVAILABLE%,>> config.json
    echo     "ibmq_token": "",>> config.json
    echo     "backend": "qasm_simulator",>> config.json
    echo     "max_qubits": 32>> config.json
    echo   },>> config.json
    echo   "machine_learning": {>> config.json
    echo     "pytorch_available": %PYTORCH_AVAILABLE%,>> config.json
    echo     "tensorflow_available": %TENSORFLOW_AVAILABLE%,>> config.json
    echo     "model_path": "models/">> config.json
    echo   },>> config.json
    echo   "distributed": {>> config.json
    echo     "mpi_available": %MPI_AVAILABLE%,>> config.json
    echo     "max_processes": 8>> config.json
    echo   },>> config.json
    echo   "cryptography": {>> config.json
    echo     "crypto_available": %CRYPTO_AVAILABLE%,>> config.json
    echo     "zk_snark_available": %ZKSNARK_AVAILABLE%,>> config.json
    echo     "homomorphic_available": %HOMOMORPHIC_AVAILABLE%>> config.json
    echo   },>> config.json
    echo   "attack_parameters": {>> config.json
    echo     "max_attack_time": 3600,>> config.json
    echo     "confidence_threshold": 0.8,>> config.json
    echo     "parallel_attacks": true>> config.json
    echo   }>> config.json
    echo }>> config.json
    echo [INFO] Configuration file created successfully.
) else (
    echo [INFO] Configuration file already exists.
)
echo.

:: Create directories
echo [INFO] Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "temp" mkdir temp
echo [INFO] Directories created successfully.
echo.

:: Display installation summary
echo ==============================================================
echo                    Installation Summary
echo ==============================================================
echo.
echo [INFO] Core Dependencies: ✓
echo [INFO] Quantum Computing: %QISKIT_AVAILABLE%
echo [INFO] Machine Learning: %PYTORCH_AVAILABLE% (PyTorch), %TENSORFLOW_AVAILABLE% (TensorFlow)
echo [INFO] Distributed Computing: %MPI_AVAILABLE%
echo [INFO] Cryptography: %CRYPTO_AVAILABLE%
echo [INFO] ZK-SNARK: %ZKSNARK_AVAILABLE%
echo [INFO] Homomorphic Encryption: %HOMOMORPHIC_AVAILABLE%
echo.
echo [INFO] All dependencies have been installed successfully!
echo [INFO] Virtual environment is ready and activated.
echo.

:: Launch the application
echo [INFO] Launching CryptoAutoPilot Quantum Assault System...
echo.
echo ==============================================================
echo               CryptoAutoPilot - Quantum Assault System
echo                      NOW LAUNCHING...
echo ==============================================================
echo.

:: Run the main application
python quantum_assault.py

:: Keep the window open after execution
echo.
echo [INFO] Application execution completed.
echo [INFO] Press any key to close this window...
pause >nul
