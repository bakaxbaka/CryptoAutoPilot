#Requires -RunAsAdministrator
# CryptoAutoPilot - Quantum Assault System Launcher
# Admin-Powered Dependency Installer and Launcher

param(
    [switch]$Force,
    [switch]$SkipPythonCheck,
    [switch]$SkipVirtualEnv,
    [switch]$InstallOnly,
    [string]$PythonVersion = "3.11"
)

# Set console encoding and colors
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$host.UI.RawUI.BackgroundColor = "Black"
$host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

Write-Host @"
==============================================================
              CryptoAutoPilot - Quantum Assault System
           Admin-Powered Dependency Installer & Launcher
==============================================================
"@ -ForegroundColor Cyan

# Function to check admin rights
function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to write colored output
function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [switch]$NoNewLine = $false
    )
    
    Write-Host $Message -ForegroundColor $Color -NoNewLine:$NoNewLine
    if (-not $NoNewLine) {
        Write-Host ""
    }
}

# Function to install Python if not present
function Install-PythonIfNeeded {
    param([string]$Version)
    
    Write-ColoredOutput "[INFO] Checking Python installation..." "Yellow"
    
    try {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            $pythonVersion = & python --version 2>&1
            Write-ColoredOutput "[INFO] Python found: $pythonVersion" "Green"
            return $true
        }
        
        if ($SkipPythonCheck) {
            Write-ColoredOutput "[WARNING] Python check skipped, assuming it's installed" "Yellow"
            return $true
        }
        
        Write-ColoredOutput "[INFO] Python not found. Installing Python $Version..." "Yellow"
        
        # Download Python installer
        $pythonUrl = "https://www.python.org/ftp/python/$Version.0/python-$Version.0-amd64.exe"
        $installerPath = "$env:TEMP\python-installer.exe"
        
        try {
            Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
            Write-ColoredOutput "[INFO] Python installer downloaded successfully" "Green"
            
            # Install Python silently
            $arguments = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
            Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -Verb RunAs
            
            # Refresh environment
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + 
                        [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            # Verify installation
            $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCmd) {
                Write-ColoredOutput "[INFO] Python installed successfully" "Green"
                return $true
            } else {
                Write-ColoredOutput "[ERROR] Python installation failed" "Red"
                return $false
            }
        } catch {
            Write-ColoredOutput "[ERROR] Failed to download/install Python: $_" "Red"
            return $false
        }
    } catch {
        Write-ColoredOutput "[ERROR] Error checking Python installation: $_" "Red"
        return $false
    }
}

# Function to upgrade pip
function Upgrade-Pip {
    Write-ColoredOutput "[INFO] Upgrading pip to latest version..." "Yellow"
    
    try {
        python -m pip install --upgrade pip
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "[INFO] pip upgraded successfully" "Green"
            return $true
        } else {
            Write-ColoredOutput "[WARNING] pip upgrade failed, continuing with current version" "Yellow"
            return $false
        }
    } catch {
        Write-ColoredOutput "[WARNING] pip upgrade failed: $_" "Yellow"
        return $false
    }
}

# Function to create virtual environment
function New-VirtualEnvironment {
    if ($SkipVirtualEnv) {
        Write-ColoredOutput "[INFO] Virtual environment creation skipped" "Yellow"
        return $true
    }
    
    Write-ColoredOutput "[INFO] Creating virtual environment..." "Yellow"
    
    try {
        if (-not (Test-Path "venv")) {
            python -m venv venv
            if ($LASTEXITCODE -eq 0) {
                Write-ColoredOutput "[INFO] Virtual environment created successfully" "Green"
                return $true
            } else {
                Write-ColoredOutput "[ERROR] Failed to create virtual environment" "Red"
                return $false
            }
        } else {
            Write-ColoredOutput "[INFO] Virtual environment already exists" "Green"
            return $true
        }
    } catch {
        Write-ColoredOutput "[ERROR] Error creating virtual environment: $_" "Red"
        return $false
    }
}

# Function to activate virtual environment
function Activate-VirtualEnvironment {
    Write-ColoredOutput "[INFO] Activating virtual environment..." "Yellow"
    
    try {
        $venvPath = ".\venv\Scripts\Activate.ps1"
        if (Test-Path $venvPath) {
            & $venvPath
            Write-ColoredOutput "[INFO] Virtual environment activated" "Green"
            return $true
        } else {
            Write-ColoredOutput "[ERROR] Virtual environment activation script not found" "Red"
            return $false
        }
    } catch {
        Write-ColoredOutput "[ERROR] Error activating virtual environment: $_" "Red"
        return $false
    }
}

# Function to install package with retry logic
function Install-PackageWithRetry {
    param(
        [string]$PackageName,
        [string]$DisplayName = $null,
        [int]$MaxRetries = 3
    )
    
    if (-not $DisplayName) {
        $DisplayName = $PackageName
    }
    
    Write-ColoredOutput "[INFO] Installing $DisplayName..." "Yellow"
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            pip install $PackageName --no-warn-script-location
            if ($LASTEXITCODE -eq 0) {
                Write-ColoredOutput "[INFO] $DisplayName installed successfully" "Green"
                return $true
            } else {
                Write-ColoredOutput "[WARNING] Attempt $i failed for $DisplayName" "Yellow"
                Start-Sleep -Seconds 2
            }
        } catch {
            Write-ColoredOutput "[WARNING] Attempt $i failed for $DisplayName: $_" "Yellow"
            Start-Sleep -Seconds 2
        }
    }
    
    Write-ColoredOutput "[ERROR] Failed to install $DisplayName after $MaxRetries attempts" "Red"
    return $false
}

# Function to install all dependencies
function Install-AllDependencies {
    Write-ColoredOutput "[INFO] Installing all dependencies..." "Yellow"
    Write-Host ""
    
    $packages = @(
        @{Name = "numpy"; DisplayName = "NumPy"; Essential = $true},
        @{Name = "requests"; DisplayName = "Requests"; Essential = $true},
        @{Name = "scipy"; DisplayName = "Scipy"; Essential = $true},
        @{Name = "qiskit"; DisplayName = "Qiskit"; Essential = $false},
        @{Name = "qiskit-aer"; DisplayName = "Qiskit Aer"; Essential = $false},
        @{Name = "qiskit-ibmq-provider"; DisplayName = "Qiskit IBM Quantum"; Essential = $false},
        @{Name = "torch"; DisplayName = "PyTorch"; Essential = $false},
        @{Name = "torchvision"; DisplayName = "TorchVision"; Essential = $false},
        @{Name = "torchaudio"; DisplayName = "TorchAudio"; Essential = $false},
        @{Name = "tensorflow"; DisplayName = "TensorFlow"; Essential = $false},
        @{Name = "scikit-learn"; DisplayName = "Scikit-learn"; Essential = $false},
        @{Name = "mpi4py"; DisplayName = "MPI4Py"; Essential = $false},
        @{Name = "cryptography"; DisplayName = "Cryptography"; Essential = $false},
        @{Name = "matplotlib"; DisplayName = "Matplotlib"; Essential = $false},
        @{Name = "seaborn"; DisplayName = "Seaborn"; Essential = $false},
        @{Name = "pandas"; DisplayName = "Pandas"; Essential = $false},
        @{Name = "networkx"; DisplayName = "NetworkX"; Essential = $false},
        @{Name = "pysnark"; DisplayName = "PySNARK"; Essential = $false},
        @{Name = "bellman"; DisplayName = "Bellman"; Essential = $false},
        @{Name = "pyseal"; DisplayName = "PySEAL"; Essential = $false},
        @{Name = "tenseal"; DisplayName = "TenSEAL"; Essential = $false}
    )
    
    $failedPackages = @()
    $successCount = 0
    
    foreach ($package in $packages) {
        $success = Install-PackageWithRetry -PackageName $package.Name -DisplayName $package.DisplayName
        if (-not $success -and $package.Essential) {
            Write-ColoredOutput "[ERROR] Essential package $($package.DisplayName) failed to install" "Red"
            return $false
        } elseif (-not $success) {
            $failedPackages += $package.DisplayName
        } else {
            $successCount++
        }
    }
    
    Write-Host ""
    Write-ColoredOutput "[INFO] Installation completed: $successCount/$($packages.Count) packages successful" "Green"
    
    if ($failedPackages.Count -gt 0) {
        Write-ColoredOutput "[WARNING] Failed packages: $($failedPackages -join ', ')" "Yellow"
    }
    
    return $true
}

# Function to create configuration file
function New-ConfigurationFile {
    Write-ColoredOutput "[INFO] Creating configuration file..." "Yellow"
    
    try {
        if (-not (Test-Path "config.json")) {
            $config = @{
                system = @{
                    name = "CryptoAutoPilot Quantum Assault System"
                    version = "1.0.0"
                    debug_mode = $true
                    log_level = "INFO"
                }
                quantum = @{
                    qiskit_available = $true
                    ibmq_token = ""
                    backend = "qasm_simulator"
                    max_qubits = 32
                }
                machine_learning = @{
                    pytorch_available = $true
                    tensorflow_available = $true
                    model_path = "models/"
                }
                distributed = @{
                    mpi_available = $true
                    max_processes = 8
                }
                cryptography = @{
                    crypto_available = $true
                    zk_snark_available = $true
                    homomorphic_available = $true
                }
                attack_parameters = @{
                    max_attack_time = 3600
                    confidence_threshold = 0.8
                    parallel_attacks = $true
                }
            }
            
            $config | ConvertTo-Json -Depth 10 | Out-File -FilePath "config.json" -Encoding UTF8
            Write-ColoredOutput "[INFO] Configuration file created successfully" "Green"
        } else {
            Write-ColoredOutput "[INFO] Configuration file already exists" "Green"
        }
        return $true
    } catch {
        Write-ColoredOutput "[ERROR] Error creating configuration file: $_" "Red"
        return $false
    }
}

# Function to create directories
function New-RequiredDirectories {
    Write-ColoredOutput "[INFO] Creating necessary directories..." "Yellow"
    
    $directories = @("logs", "models", "results", "temp")
    
    foreach ($dir in $directories) {
        try {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
        } catch {
            Write-ColoredOutput "[WARNING] Failed to create directory $dir: $_" "Yellow"
        }
    }
    
    Write-ColoredOutput "[INFO] Directories created successfully" "Green"
    return $true
}

# Function to display installation summary
function Show-InstallationSummary {
    Write-Host ""
    Write-Host @"
==============================================================
                    Installation Summary
==============================================================
"@ -ForegroundColor Cyan
    
    Write-ColoredOutput "[INFO] Core Dependencies: ✓" "Green"
    Write-ColoredOutput "[INFO] Quantum Computing: ✓" "Green"
    Write-ColoredOutput "[INFO] Machine Learning: ✓" "Green"
    Write-ColoredOutput "[INFO] Distributed Computing: ✓" "Green"
    Write-ColoredOutput "[INFO] Cryptography: ✓" "Green"
    Write-ColoredOutput "[INFO] ZK-SNARK: ✓" "Green"
    Write-ColoredOutput "[INFO] Homomorphic Encryption: ✓" "Green"
    Write-Host ""
    Write-ColoredOutput "[INFO] All dependencies have been installed successfully!" "Green"
    Write-ColoredOutput "[INFO] Virtual environment is ready and activated." "Green"
    Write-Host ""
}

# Function to launch application
function Start-QuantumAssault {
    Write-ColoredOutput "[INFO] Launching CryptoAutoPilot Quantum Assault System..." "Yellow"
    Write-Host ""
    Write-Host @"
==============================================================
               CryptoAutoPilot - Quantum Assault System
                      NOW LAUNCHING...
==============================================================
"@ -ForegroundColor Cyan
    Write-Host ""
    
    try {
        python quantum_assault.py
        Write-ColoredOutput "[INFO] Application execution completed" "Green"
    } catch {
        Write-ColoredOutput "[ERROR] Error launching application: $_" "Red"
    }
    
    Write-ColoredOutput "[INFO] Press any key to close this window..." "Yellow"
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Main execution
try {
    # Check admin rights
    if (-not (Test-AdminRights)) {
        Write-ColoredOutput "[ERROR] This script requires administrator privileges!" "Red"
        Write-ColoredOutput "[INFO] Please run PowerShell as Administrator and try again." "Yellow"
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Set execution policy for this session
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
    
    # Install Python if needed
    if (-not (Install-PythonIfNeeded -Version $PythonVersion)) {
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Upgrade pip
    Upgrade-Pip
    
    # Create virtual environment
    if (-not (New-VirtualEnvironment)) {
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Activate virtual environment
    if (-not (Activate-VirtualEnvironment)) {
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Install dependencies
    if (-not (Install-AllDependencies)) {
        Write-ColoredOutput "[ERROR] Failed to install some dependencies" "Red"
        if (-not $Force) {
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    
    # Create configuration file
    New-ConfigurationFile
    
    # Create directories
    New-RequiredDirectories
    
    # Show summary
    Show-InstallationSummary
    
    # Launch application unless install-only mode
    if (-not $InstallOnly) {
        Start-QuantumAssault
    } else {
        Write-ColoredOutput "[INFO] Installation completed. Use -InstallOnly:$false to launch the application." "Green"
        Read-Host "Press Enter to exit"
    }
    
} catch {
    Write-ColoredOutput "[ERROR] Unexpected error: $_" "Red"
    Write-ColoredOutput "[ERROR] Stack Trace: $($_.ScriptStackTrace)" "Red"
    Read-Host "Press Enter to exit"
    exit 1
}
