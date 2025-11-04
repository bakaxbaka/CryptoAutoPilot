#!/bin/bash
# CryptoAutoPilot - Simple Launcher Script
# This script installs core dependencies and launches the application

echo "=============================================================="
echo "       CryptoAutoPilot - Bitcoin Vulnerability Scanner"
echo "                    Quick Launcher"
echo "=============================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "[INFO] Python 3 found: $(python3 --version)"
echo ""

# Install core dependencies
echo "[INFO] Installing core dependencies..."
echo "[INFO] This may take a few minutes on first run..."
echo ""

pip3 install -q Flask Flask-SQLAlchemy SQLAlchemy requests ecdsa base58 numpy scipy gunicorn python-dotenv 2>&1 | grep -v "Requirement already satisfied" || true

if [ $? -eq 0 ]; then
    echo "[INFO] Core dependencies installed successfully!"
else
    echo "[WARNING] Some dependencies may have failed to install, but continuing..."
fi

echo ""
echo "[INFO] Initializing database..."
python3 -c "from main import app, db; app.app_context().push(); db.create_all(); print('[INFO] Database initialized successfully!')" 2>&1

echo ""
echo "=============================================================="
echo "              Launching CryptoAutoPilot..."
echo "=============================================================="
echo ""
echo "[INFO] The application will be available at:"
echo "       http://localhost:5000"
echo ""
echo "[INFO] Press CTRL+C to stop the server"
echo ""

# Launch the application
python3 main.py
