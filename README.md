# CryptoAutoPilot - Bitcoin Vulnerability Scanner

## Overview

CryptoAutoPilot is a comprehensive Bitcoin vulnerability scanner that analyzes ECDSA signature weaknesses in Bitcoin transactions. The application searches for cryptographic vulnerabilities that could lead to private key recovery, including k-reuse attacks, weak random number generation, brain wallet vulnerabilities, and various other ECDSA implementation flaws.

## 🚀 Quick Start

### One-Click Launch

**Windows Users:** Simply double-click `start_app.bat` and the application will:

- Automatically install all required dependencies
- Set up a virtual environment
- Launch the web server
- Open your browser to the application

The application will be available at `http://localhost:5000`

### Manual Launch

```bash
# Clone and navigate to the directory
git clone <repository-url>
cd CryptoAutoPilot

# Run the launcher (Windows)
start_app.bat

# Or run manually (All platforms)
pip install -r requirements.txt
python main.py
```

## Features

### Core Vulnerability Detection

- **K-reuse Detection**: Identifies repeated nonce values across multiple signatures
- **Weak R-value Analysis**: Detects signatures with abnormally small R values
- **LSB/MSB Bias Detection**: Identifies biases in signature generation
- **Lattice Attack Preparation**: Prepares data for advanced lattice-based attacks
- **Brain Wallet Vulnerability Scanning**: Detects weak private key patterns

### Advanced Analysis Engine

- **Real-time Block Analysis**: Analyzes Bitcoin blocks as they are mined
- **Autopilot Mode**: Automated sequential block scanning with configurable parameters
- **Multi-threaded Processing**: Parallel transaction analysis for improved performance
- **Private Key Recovery**: Mathematical recovery of private keys from vulnerable signatures
- **Address Generation**: Generates multiple Bitcoin address formats from recovered keys

### Web Interface

- **Real-time Dashboard**: Live vulnerability statistics and analysis progress
- **Interactive Controls**: Start/stop analysis, configure parameters
- **Detailed Reporting**: Comprehensive vulnerability reports with recovery details
- **Transaction Explorer**: Integration with blockchain.com for transaction verification
- **Responsive Design**: Bootstrap-based terminal/hacker aesthetic

## System Architecture

### Frontend

- **Framework**: Flask web application with Bootstrap 5.1.3
- **Templates**: Jinja2 templating engine
- **JavaScript**: Vanilla JS for real-time updates and AJAX communication
- **Styling**: Custom CSS with terminal-style green-on-black color scheme

### Backend

- **Framework**: Flask (Python 3.11+)
- **Database**: SQLite with SQLAlchemy ORM (PostgreSQL-ready)
- **Concurrency**: ThreadPoolExecutor for parallel processing
- **API Integration**: Blockstream API for Bitcoin blockchain data
- **Cryptographic Libraries**: ECDSA, Base58, Hashlib for Bitcoin operations

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space

### Installation Methods

#### Method 1: Automated Launcher (Recommended)

**For Windows Users:**
1. Download or clone the repository
2. Navigate to the `CryptoAutoPilot` directory
3. Double-click `start_app.bat`
4. Wait for automatic setup (first run only)
5. The application will open in your browser

**Features:**

- ✅ Automatic dependency installation
- ✅ Virtual environment setup
- ✅ Web server launch
- ✅ Browser integration
- ✅ Error handling and recovery

#### Method 2: Manual Installation

**For All Platforms:**

1. **Clone the repository:**

```bash
git clone <repository-url>
cd CryptoAutoPilot
```

2. **Install dependencies:**

```bash
# Using requirements.txt
pip install -r requirements.txt

# Or install core packages manually
pip install flask flask-sqlalchemy python-dotenv numpy requests scipy
pip install torch tensorflow scikit-learn cryptography matplotlib pandas networkx
```

3. **Set up virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Initialize the database:**

```bash
python -c "from main import app, db; app.app_context().push(); db.create_all()"
```

5. **Launch the application:**

```bash
python main.py
```

The application will be available at `http://localhost:5000`

### Launcher Files

| File | Purpose | Platform |
|------|---------|----------|
| `start_app.bat` | Simple one-click launcher | Windows |
| `start_admin.bat` | Admin-powered launcher with advanced features | Windows |
| `launch_admin.bat` | Advanced launcher with multiple fallback methods | Windows |
| `start.ps1` | PowerShell launcher with admin capabilities | Windows |

### Troubleshooting

**Common Issues:**

1. **Python not found:**

   - Install Python 3.8+ from [python.org](https://python.org)
   - Ensure "Add Python to PATH" is checked during installation

2. **Permission errors:**

   - Use `start_admin.bat` for administrator privileges
   - Or run Command Prompt as administrator

3. **Port already in use:**

   - Close other applications using port 5000
   - The application will show an error if the port is unavailable

4. **Package installation fails:**

   - Ensure internet connection is stable
   - Try running the launcher again (it has retry logic)
   - Check Python and pip are working correctly

## Usage

### Getting Started

1. **Launch the application:**

   - Windows: Double-click `start_app.bat`
   - Manual: `python main.py`

2. **Open your browser:** Navigate to `http://localhost:5000`

3. **Explore the dashboard:**

   - View real-time vulnerability statistics
   - Monitor analysis progress
   - Access control panels

### Web Interface Guide

#### 🎯 Main Dashboard

- **Live Statistics**: Real-time vulnerability counts and analysis progress
- **Quick Actions**: Start/stop autopilot, analyze specific blocks
- **Recent Activity**: Latest vulnerability discoveries and analysis results

#### 🔍 Block Analysis
1. **Enter Block Information**:

   - Block number (e.g., 800000)
   - Block hash (e.g., 000000000000000000076c036ff5119e5a5a74df77abf64203473364509f7732)

2. **Start Analysis**:

   - Click "Analyze Block"
   - Monitor progress in real-time
   - View detailed results when complete

3. **Review Results**:

   - Vulnerability types discovered
   - Affected transactions
   - Private key recovery attempts
   - Risk assessment

#### 🤖 Autopilot Mode
1. **Configure Settings**:

   - **Direction**: Forward (newer blocks) or Backward (older blocks)
   - **Delay**: Time between block analyses (1-60 seconds)
   - **Range**: Number of blocks to analyze

2. **Start Automated Scanning**:

   - Click "Start Autopilot"
   - Monitor continuous analysis
   - View vulnerability discoveries in real-time

3. **Control and Monitor**:

   - Pause/resume scanning
   - Change direction on the fly
   - View cumulative statistics
   - Export results

#### 🧪 Testing and Development

**Manual Recovery Testing:**
- Navigate to `/test_manual_recovery`
- Test k-reuse recovery algorithms
- Validate results with real transaction data
- Debug and optimize recovery methods

#### Configuration

- Edit `config.json` for custom settings
- Modify analysis parameters
- Configure API endpoints and timeouts

### Advanced Features

#### 🔐 Quantum-Enhanced Analysis

The system includes quantum computing algorithms for:

- **Shor's Algorithm**: Factor large numbers for RSA attacks
- **Grover's Algorithm**: Quadratic speedup for brute force searches
- **Quantum Phase Estimation**: Enhanced eigenvalue extraction

#### 🧠 Machine Learning Integration

- **LSTM Networks**: Predict vulnerability patterns
- **CNN Models**: Analyze signature structures
- **Random Forest**: Classify risk levels
- **Neural Networks**: Optimize attack strategies

#### 🌐 Distributed Computing

- **MPI Support**: Parallel processing across multiple machines
- **Load Balancing**: Optimize resource utilization
- **Fault Tolerance**: Handle node failures gracefully

#### 🛡️ Zero-Knowledge Proofs

- **ZK-SNARK Integration**: Generate privacy-preserving proofs
- **Homomorphic Encryption**: Process encrypted data
- **Secure Multi-Party Computation**: Collaborative analysis

## API Endpoints

### Core Analysis

- `POST /analyze_block`: Analyze a specific Bitcoin block
- `GET /autopilot_status`: Get current autopilot status
- `POST /start_autopilot`: Start automated block scanning
- `POST /stop_autopilot`: Stop automated scanning

### Data Retrieval
- `GET /vulnerability_stats`: Get vulnerability statistics
- `GET /export/<analysis_key>`: Export analysis results
- `GET /test_manual_recovery`: Test manual recovery with real data

### Configuration
- `GET /config`: Get current configuration
- `POST /config`: Update configuration settings

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'development' for debug mode
- `DATABASE_URL`: Database connection string (default: SQLite)
- `BLOCKSTREAM_API`: Blockstream API endpoint
- `SESSION_SECRET`: Flask session secret key

### Application Settings
- `BLOCKSTREAM_API`: Bitcoin blockchain API endpoint
- `MAX_CONCURRENT_ANALYSIS`: Maximum concurrent analysis threads
- `ANALYSIS_TIMEOUT`: Timeout for individual analysis operations
- `AUTOPLOT_DELAY`: Delay between autopilot block analyses

## Vulnerability Detection Methods

### K-reuse Attack
The system detects when the same nonce (k-value) is used across multiple ECDSA signatures. This vulnerability allows mathematical recovery of the private key using the formula:

```
k = (z₁ - z₂) / (s₁ - s₂) mod n
d = (s₁ × k - z₁) / r mod n
```

### Weak R-value Analysis
Identifies signatures with unusually small R values, which can indicate poor random number generation or potential vulnerabilities in the signing process.

### Brain Wallet Detection
Scans for private keys generated from weak patterns such as:
- Leading zeros
- Repeating characters
- Common test patterns
- Sequential patterns
- Low entropy keys

### Lattice Attack Preparation
Collects and prepares signature data for advanced lattice-based cryptographic attacks, which can recover private keys from signatures with biased nonce generation.

## Security Considerations

### Data Privacy
- All recovered private keys are handled securely
- No sensitive data is stored in plain text
- Transactions are analyzed anonymously

### API Security
- Rate limiting for external API calls
- Input validation for all user inputs
- Secure session management
- Protection against common web vulnerabilities

### Operational Security
- No logging of recovered private keys
- Secure handling of cryptographic materials
- Regular security updates and dependency management

## Development

### Project Structure
```
CryptoAutoPilot/
├── main.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Main web interface
├── static/
│   ├── app.js            # Frontend JavaScript
│   └── style.css         # Custom CSS styles
└── README.md             # This documentation
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Deployment

### Production Deployment
1. Set up a production database (PostgreSQL recommended)
2. Configure environment variables
3. Use Gunicorn as WSGI server:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
```

### Replit Deployment
The application is configured for easy deployment on Replit:
1. Import the project to Replit
2. Configure the Nix packages (PostgreSQL, OpenSSL, etc.)
3. Set up the run command with Gunicorn
4. Deploy using Replit's autoscale infrastructure

## Troubleshooting

### Common Issues

**Database Connection Errors**
- Ensure SQLite file permissions are correct
- Check DATABASE_URL environment variable
- Verify database initialization

**API Rate Limiting**
- Blockstream API has rate limits
- Implement proper request throttling
- Consider using multiple API endpoints

**Memory Usage**
- Large block analysis can consume significant memory
- Adjust MAX_CONCURRENT_ANALYSIS setting
- Monitor system resources during operation

**Performance Issues**
- Reduce analysis thread count for slower systems
- Increase AUTOPLOT_DELAY for less frequent scanning
- Consider using SSD storage for database

## License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations when using cryptographic analysis tools.

## Disclaimer

This tool is designed for educational and research purposes. Users are responsible for ensuring compliance with applicable laws and regulations. The developers are not responsible for any misuse of this software.

## Changelog

### June 21, 2025
- Initial setup and basic vulnerability detection
- Enhanced k-reuse detection with block-wide R-value analysis
- Added WIF private key recovery functionality
- Implemented transaction explorer integration
- Enhanced signature extraction with improved DER parsing
- Added raw transaction parsing with proper SIGHASH calculation
- Replaced all mock data with real transaction fetching
- Implemented dynamic weak key detection
- Added comprehensive Bitcoin address generation

## Support

For issues, questions, or contributions, please open an issue on the project repository or contact the development team.
