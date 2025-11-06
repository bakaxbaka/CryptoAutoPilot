# Quick Start Guide - CryptoAutoPilot

## ✅ Code Status: FIXED AND READY

All critical issues have been resolved. The application is ready to run!

## 🚀 Start the Application

### Option 1: Direct Start (Fastest)
```bash
python3 main.py
```

Then open your browser to: **http://localhost:5000**

### Option 2: With Virtual Environment (Recommended for Production)
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install core dependencies
pip install Flask Flask-SQLAlchemy requests ecdsa base58 python-dotenv

# Run the application
python3 main.py
```

### Option 3: Windows One-Click Launcher
```bash
start_app.bat
```

## 📋 What Was Fixed

1. **requirements.txt** - Removed invalid `hashlib` package (it's built-in)
2. **main.py** - Fixed indentation errors in route functions
3. **Database** - Verified initialization works correctly
4. **All Python files** - Validated syntax (all pass)

## 🧪 Test the Application

Run the test script to verify everything works:
```bash
python3 test_app.py
```

Expected output:
```
✓ Flask app loaded successfully
✓ All routes registered
✓ Application is ready to run!
```

## 📊 Features Available

- **Block Analysis** - Analyze Bitcoin blocks for vulnerabilities
- **Autopilot Mode** - Automated sequential block scanning
- **K-reuse Detection** - Identify repeated nonce values
- **Private Key Recovery** - Mathematical recovery from vulnerable signatures
- **Real-time Dashboard** - Live vulnerability statistics
- **Export Results** - Download analysis data as JSON

## 🔧 Configuration

Edit `config.py` or set environment variables:

```bash
export BLOCKSTREAM_API="https://blockstream.info/api"
export REQUEST_TIMEOUT="15"
export MAX_CONCURRENT_ANALYSIS="4"
```

## 📁 Project Structure

```
/vercel/sandbox/
├── main.py                    # Main Flask application ✅
├── config.py                  # Configuration settings ✅
├── requirements.txt           # Python dependencies ✅
├── templates/                 # HTML templates ✅
│   ├── index.html
│   ├── analysis_detail.html
│   └── vulnerability_type.html
├── static/                    # CSS/JS files
├── instance/                  # Database location
│   └── bitcoin_vulnerabilities.db
└── tests/                     # Test files ✅
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find and kill process using port 5000
lsof -ti:5000 | xargs kill -9
```

### Missing Dependencies
```bash
pip install Flask Flask-SQLAlchemy requests ecdsa base58 python-dotenv
```

### Database Issues
```bash
# Reinitialize database
python3 -c "from main import app, db; app.app_context().push(); db.create_all()"
```

## 📚 API Endpoints

- `GET  /` - Main dashboard
- `POST /analyze` - Analyze a block
- `POST /autopilot/start` - Start autopilot
- `GET  /autopilot/status` - Get autopilot status
- `GET  /api/dashboard_stats` - Get statistics
- `GET  /export/<key>` - Export analysis results

## 🎯 Next Steps

1. Start the application: `python3 main.py`
2. Open browser to http://localhost:5000
3. Enter a Bitcoin block number (e.g., 800000)
4. Click "Analyze Block"
5. View vulnerability results

## 💡 Tips

- Start with recent blocks (800000+) for faster analysis
- Use autopilot mode for continuous scanning
- Export results for offline analysis
- Check the dashboard for real-time statistics

## 📞 Support

For issues or questions:
- Check `FIXES_APPLIED.md` for detailed fix information
- Review `README.md` for comprehensive documentation
- Check logs in the console output

---

**Status:** ✅ All systems operational - Ready to scan!
