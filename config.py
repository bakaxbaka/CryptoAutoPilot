import os
from typing import Dict, Any

class Config:
    """Configuration management for the Bitcoin Vulnerability Scanner"""
    
    # API Configuration
    BLOCKSTREAM_API = os.getenv('BLOCKSTREAM_API', 'https://blockstream.info/api')
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '15'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', '0.1'))
    
    # Blockchain Explorer URLs
    BLOCKCHAIN_EXPLORERS = {
        'blockchain.com': {
            'tx': os.getenv('BLOCKCHAIN_COM_TX_URL', 'https://www.blockchain.com/explorer/transactions/btc/'),
            'address': os.getenv('BLOCKCHAIN_COM_ADDRESS_URL', 'https://www.blockchain.com/explorer/addresses/btc/'),
            'block': os.getenv('BLOCKCHAIN_COM_BLOCK_URL', 'https://www.blockchain.com/explorer/blocks/btc/')
        },
        'blockstream.info': {
            'tx': os.getenv('BLOCKSTREAM_INFO_TX_URL', 'https://blockstream.info/tx/'),
            'address': os.getenv('BLOCKSTREAM_INFO_ADDRESS_URL', 'https://blockstream.info/address/'),
            'block': os.getenv('BLOCKSTREAM_INFO_BLOCK_URL', 'https://blockstream.info/block/')
        },
        'mempool.space': {
            'tx': os.getenv('MEMPOOL_SPACE_TX_URL', 'https://mempool.space/tx/'),
            'address': os.getenv('MEMPOOL_SPACE_ADDRESS_URL', 'https://mempool.space/address/'),
            'block': os.getenv('MEMPOOL_SPACE_BLOCK_URL', 'https://mempool.space/block/')
        }
    }
    
    # Default explorer to use
    DEFAULT_EXPLORER = os.getenv('DEFAULT_EXPLORER', 'blockstream.info')
    
    # Application Configuration
    MAX_CONCURRENT_ANALYSIS = int(os.getenv('MAX_CONCURRENT_ANALYSIS', '4'))
    ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', '30'))
    AUTOPLOT_DELAY = float(os.getenv('AUTOPLOT_DELAY', '1.0'))
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///bitcoin_vulnerabilities.db')
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'bitcoin_scanner.log')
    
    @classmethod
    def get_explorer_url(cls, explorer: str = None, resource_type: str = 'tx') -> str:
        """Get the configured explorer URL for a specific resource type"""
        if explorer is None:
            explorer = cls.DEFAULT_EXPLORER
        
        if explorer not in cls.BLOCKCHAIN_EXPLORERS:
            explorer = cls.DEFAULT_EXPLORER
        
        return cls.BLOCKCHAIN_EXPLORERS[explorer].get(resource_type, '')
    
    @classmethod
    def get_all_explorers(cls) -> Dict[str, Dict[str, str]]:
        """Get all configured blockchain explorers"""
        return cls.BLOCKCHAIN_EXPLORERS
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check API URL
        if not cls.BLOCKSTREAM_API.startswith('http'):
            issues.append('BLOCKSTREAM_API must be a valid URL')
        
        # Check timeouts
        if cls.REQUEST_TIMEOUT < 1:
            issues.append('REQUEST_TIMEOUT must be at least 1 second')
        
        if cls.ANALYSIS_TIMEOUT < 1:
            issues.append('ANALYSIS_TIMEOUT must be at least 1 second')
        
        # Check concurrent analysis
        if cls.MAX_CONCURRENT_ANALYSIS < 1:
            issues.append('MAX_CONCURRENT_ANALYSIS must be at least 1')
        
        # Check explorer configuration
        if cls.DEFAULT_EXPLORER not in cls.BLOCKCHAIN_EXPLORERS:
            issues.append(f'DEFAULT_EXPLORER "{cls.DEFAULT_EXPLORER}" is not configured')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
