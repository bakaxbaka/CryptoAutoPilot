# Bitcoin Vulnerability Scanner

## Overview

This is a comprehensive Bitcoin vulnerability scanner that analyzes ECDSA signature weaknesses in Bitcoin transactions. The application searches for cryptographic vulnerabilities that could lead to private key recovery, including k-reuse attacks, weak random number generation, brain wallet vulnerabilities, and various other ECDSA implementation flaws.

## System Architecture

### Frontend Architecture
- **Framework**: Flask web application with Bootstrap-based terminal-style UI
- **Templates**: Jinja2 templates with responsive design
- **JavaScript**: Vanilla JS for real-time updates and AJAX communication
- **Styling**: Custom CSS with terminal/hacker aesthetic using green-on-black color scheme

### Backend Architecture
- **Framework**: Flask (Python 3.11+)
- **WSGI Server**: Gunicorn with auto-scaling deployment
- **Database**: SQLite with SQLAlchemy ORM
- **Concurrency**: ThreadPoolExecutor for parallel transaction analysis
- **APIs**: Integration with Blockstream API for Bitcoin blockchain data

### Core Analysis Engine
- **ECDSA Analysis**: Custom implementation using `ecdsa` library
- **Cryptographic Methods**: Advanced mathematical algorithms for vulnerability detection
- **Pattern Recognition**: Machine learning-like pattern detection for vulnerability hotspots
- **Recovery Algorithms**: Multiple private key recovery methods including k-reuse exploitation

## Key Components

### 1. Block Analyzer (`main.py`)
- Fetches Bitcoin block data from Blockstream API
- Parses transactions and extracts ECDSA signatures
- Implements multiple vulnerability detection algorithms:
  - K-reuse detection with automatic private key recovery
  - Weak R-value analysis
  - LSB/MSB bias detection
  - Lattice attack preparation
  - Brain wallet vulnerability scanning

### 2. Database Models
- **AnalysisResult**: Stores block analysis metadata and status
- **Vulnerabilities**: Detailed vulnerability findings with risk levels
- Uses SQLite for development with PostgreSQL readiness

### 3. Autopilot Mode
- Automated sequential block scanning
- Directional scanning (forward/backward)
- Configurable delays and ranges
- Real-time progress tracking
- Smart targeting based on historical vulnerability patterns

### 4. Vulnerability Detection Modules
- **CryptoDeep Integration**: Advanced cryptographic analysis methods
- **Known Vulnerable Keys**: Database of previously compromised keys
- **Pattern Analysis**: Historical vulnerability hotspot identification
- **Private Key Validation**: Comprehensive key security assessment

### 5. Web Interface
- Real-time dashboard with vulnerability statistics
- Interactive analysis controls
- Detailed vulnerability reporting
- Copy-to-clipboard functionality for transaction IDs
- Bootstrap-based responsive design

## Data Flow

1. **User Input**: Block number or hash submitted via web interface
2. **API Fetch**: Blockstream API retrieval of block and transaction data
3. **Signature Extraction**: Parse DER-encoded signatures from transaction scripts
4. **Vulnerability Analysis**: Multi-threaded analysis using various detection algorithms
5. **Database Storage**: Results stored in SQLite with detailed metadata
6. **Real-time Updates**: WebSocket-like updates to frontend dashboard
7. **Report Generation**: Comprehensive vulnerability reports with recovery details

## External Dependencies

### Core Libraries
- **Flask**: Web framework and routing
- **SQLAlchemy**: Database ORM and migrations
- **ecdsa**: Elliptic curve cryptography operations
- **requests**: HTTP client for blockchain API calls
- **numpy/scipy**: Mathematical operations for cryptographic analysis

### Blockchain APIs
- **Blockstream API**: Primary source for Bitcoin blockchain data
- **Fallback APIs**: Multiple API endpoints for redundancy
- **Rate Limiting**: Automatic request throttling and retry logic

### Frontend Dependencies
- **Bootstrap 5.1.3**: UI framework
- **Font Awesome 6.0**: Icon library
- **Custom CSS**: Terminal-style theming

## Deployment Strategy

### Replit Deployment
- **Target**: Autoscale deployment on Replit infrastructure
- **WSGI**: Gunicorn with optimized worker configuration
- **Port Configuration**: 5000 internal, 80 external
- **Process Management**: Auto-restart and health monitoring

### Environment Configuration
- **Nix Packages**: PostgreSQL, OpenSSL, pkg-config, and crypto libraries
- **Python**: 3.11 with comprehensive dependency management
- **Database**: SQLite for development, PostgreSQL-ready for production

### Security Considerations
- Session secret key from environment variables
- Input validation for all user inputs
- Rate limiting for API calls
- Secure handling of recovered private keys

## Changelog

- June 21, 2025. Initial setup
- June 21, 2025. Enhanced k-reuse detection with block-wide R-value analysis and WIF private key recovery
- June 21, 2025. Added transaction explorer integration with blockchain.com redirect functionality  
- June 21, 2025. Implemented comprehensive signature extraction with improved DER parsing
- June 21, 2025. Added raw transaction parsing with proper Bitcoin SIGHASH calculation for accurate vulnerability detection

## User Preferences

Preferred communication style: Simple, everyday language.