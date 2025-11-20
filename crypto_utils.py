#!/usr/bin/env python3
"""
Cryptographic Utilities Module
Uses well-tested libraries for ECC operations and key conversions
"""

import hashlib
import logging
import math
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from ecdsa import SECP256k1, SigningKey, VerifyingKey
    from ecdsa import ellipticcurve, numbertheory
    from ecdsa.util import sigdecode_der
    ECDSA_AVAILABLE = True
except ImportError:
    ECDSA_AVAILABLE = False
    logging.warning("ecdsa library not available, using fallback implementations")

try:
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography library not available, using fallback implementations")

try:
    import base58
    BASE58_AVAILABLE = True
except ImportError:
    BASE58_AVAILABLE = False
    logging.warning("base58 library not available, using fallback implementations")

@dataclass
class CryptoResult:
    """Result of cryptographic operation"""
    success: bool
    data: Any = None
    error: str = ""
    operation: str = ""

class SecureCryptoUtils:
    """Secure cryptographic utilities using well-tested libraries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.attack_id = None
        
        # Validate library availability
        self._validate_libraries()
    
    def _validate_libraries(self):
        """Validate that required cryptographic libraries are available"""
        missing_libs = []
        
        if not ECDSA_AVAILABLE:
            missing_libs.append("ecdsa")
        
        if not CRYPTOGRAPHY_AVAILABLE:
            missing_libs.append("cryptography")
        
        if not BASE58_AVAILABLE:
            missing_libs.append("base58")
        
        if missing_libs:
            self.logger.error(f"Missing required libraries: {missing_libs}")
            raise ImportError(f"Missing required libraries: {missing_libs}")
    
    def set_attack_id(self, attack_id: str):
        """Set attack ID for logging and tracking"""
        self.attack_id = attack_id
        self.logger.info(f"Attack ID set: {attack_id}")
    
    def private_key_to_public_key(self, private_key: int, compressed: bool = True) -> Optional[str]:
        """
        Convert private key to public key using ecdsa library
        Returns public key in hex format using the requested SEC1 encoding
        """
        try:
            if not ECDSA_AVAILABLE:
                raise RuntimeError("ecdsa library not available")
            
            # Convert private key to bytes
            private_key_bytes = private_key.to_bytes(32, byteorder='big')
            
            # Create signing key
            signing_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            
            # Get verifying key (public key)
            verifying_key = signing_key.get_verifying_key()
            
            # Get public key bytes in the requested format
            encoding = "compressed" if compressed else "uncompressed"
            public_key_bytes = verifying_key.to_string(encoding)
            
            # Convert to hex
            public_key_hex = public_key_bytes.hex()
            
            self.logger.debug(f"Generated public key: {public_key_hex}")
            return public_key_hex
            
        except Exception as e:
            self.logger.error(f"Error converting private key to public key: {e}")
            return None
    
    def public_key_to_coordinates(self, public_key_hex: str) -> Optional[Tuple[int, int]]:
        """
        Convert public key hex to (x, y) coordinates
        """
        try:
            if not ECDSA_AVAILABLE:
                raise RuntimeError("ecdsa library not available")

            # Convert hex to bytes
            public_key_bytes = bytes.fromhex(public_key_hex)

            verifying_key = self._verifying_key_from_bytes(public_key_bytes)

            point = verifying_key.pubkey.point

            return (point.x(), point.y())

        except Exception as e:
            self.logger.error(f"Error converting public key to coordinates: {e}")
            return None

    def _decode_public_key_bytes(self, public_key_bytes: bytes) -> Tuple[int, int]:
        """Decode raw, compressed or uncompressed public key bytes into coordinates."""
        length = len(public_key_bytes)

        if length == 33:
            return self._decode_compressed_key(public_key_bytes)
        if length == 65:
            if public_key_bytes[0] != 0x04:
                raise ValueError("Invalid uncompressed public key prefix")
            x = int.from_bytes(public_key_bytes[1:33], "big")
            y = int.from_bytes(public_key_bytes[33:], "big")
            return x, y
        if length == 64:
            x = int.from_bytes(public_key_bytes[:32], "big")
            y = int.from_bytes(public_key_bytes[32:], "big")
            return x, y

        raise ValueError(f"Unsupported public key length: {length}")

    def _decode_compressed_key(self, public_key_bytes: bytes) -> Tuple[int, int]:
        """Decode a compressed SEC1 public key into affine coordinates."""
        prefix = public_key_bytes[0]
        if prefix not in (0x02, 0x03):
            raise ValueError("Invalid compressed public key prefix")

        x = int.from_bytes(public_key_bytes[1:], "big")
        curve = SECP256k1.curve
        p = curve.p()

        # y^2 = x^3 + ax + b (mod p)
        alpha = (pow(x, 3, p) + curve.a() * x + curve.b()) % p
        beta = numbertheory.square_root_mod_prime(alpha, p)

        if (beta % 2) != (prefix % 2):
            beta = (-beta) % p

        return x, beta
    
    def private_key_to_wif(self, private_key: int, compressed: bool = True) -> Optional[str]:
        """
        Convert private key to WIF format using proper encoding
        """
        try:
            # Convert private key to bytes
            private_key_bytes = private_key.to_bytes(32, byteorder='big')
            
            # Add prefix byte (0x80 for mainnet)
            extended_key = b'\x80' + private_key_bytes
            
            # Add suffix byte for compressed keys
            if compressed:
                extended_key += b'\x01'
            
            # Double SHA-256 hash for checksum
            first_hash = hashlib.sha256(extended_key).digest()
            second_hash = hashlib.sha256(first_hash).digest()
            
            # Add checksum (first 4 bytes)
            checksum = second_hash[:4]
            final_key = extended_key + checksum
            
            # Base58 encode
            if BASE58_AVAILABLE:
                wif_key = base58.b58encode(final_key).decode('utf-8')
            else:
                wif_key = self._base58_encode_fallback(final_key)
            
            self.logger.debug(f"Generated WIF key: {wif_key[:10]}...")
            return wif_key
            
        except Exception as e:
            self.logger.error(f"Error converting private key to WIF: {e}")
            return None
    
    def wif_to_private_key(self, wif_key: str) -> Optional[int]:
        """
        Convert WIF key to private key integer
        """
        try:
            # Base58 decode
            if BASE58_AVAILABLE:
                decoded = base58.b58decode(wif_key)
            else:
                decoded = self._base58_decode_fallback(wif_key)
            
            # Remove prefix (0x80) and checksum (last 4 bytes)
            # Also remove compression suffix (0x01) if present
            if len(decoded) == 38:  # Compressed
                private_key_bytes = decoded[1:-5]
            elif len(decoded) == 37:  # Uncompressed
                private_key_bytes = decoded[1:-4]
            else:
                raise ValueError(f"Invalid WIF key length: {len(decoded)}")
            
            # Convert to integer
            private_key = int.from_bytes(private_key_bytes, byteorder='big')
            
            self.logger.debug(f"Decoded private key from WIF")
            return private_key
            
        except Exception as e:
            self.logger.error(f"Error converting WIF to private key: {e}")
            return None
    
    def wif_to_public_key(self, wif_key: str) -> Optional[str]:
        """
        Convert WIF key directly to public key
        """
        try:
            private_key = self.wif_to_private_key(wif_key)
            if private_key is None:
                return None

            decoded = base58.b58decode(wif_key) if BASE58_AVAILABLE else self._base58_decode_fallback(wif_key)
            compressed = len(decoded) == 38

            return self.private_key_to_public_key(private_key, compressed=compressed)
            
        except Exception as e:
            self.logger.error(f"Error converting WIF to public key: {e}")
            return None
    
    def parse_signature_der(self, signature_der: str) -> Optional[Tuple[int, int]]:
        """
        Parse DER-encoded signature using cryptography library
        Returns (r, s) tuple
        """
        try:
            signature_bytes = bytes.fromhex(signature_der)

            if CRYPTOGRAPHY_AVAILABLE:
                r, s = decode_dss_signature(signature_bytes)
            else:
                r, s = sigdecode_der(signature_bytes, SECP256k1.order)
            
            self.logger.debug(f"Parsed DER signature: r={r}, s={s}")
            return (r, s)
            
        except Exception as e:
            self.logger.error(f"Error parsing DER signature: {e}")
            return None
    
    def verify_signature(self, public_key_hex: str, signature_der: str, message_hash: bytes) -> bool:
        """
        Verify ECDSA signature using ecdsa library
        """
        try:
            if not ECDSA_AVAILABLE:
                raise RuntimeError("ecdsa library not available")
            
            public_key_bytes = bytes.fromhex(public_key_hex)
            verifying_key = self._verifying_key_from_bytes(public_key_bytes)

            signature_bytes = bytes.fromhex(signature_der)

            is_valid = verifying_key.verify_digest(signature_bytes, message_hash, sigdecode=sigdecode_der)
            
            self.logger.debug(f"Signature verification result: {is_valid}")
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error verifying signature: {e}")
            return False
    
    def point_multiply(self, x: int, y: int, scalar: int) -> Optional[Tuple[int, int]]:
        """
        Multiply point by scalar using ecdsa library
        Returns (x, y) coordinates or None for point at infinity
        """
        try:
            if not ECDSA_AVAILABLE:
                raise RuntimeError("ecdsa library not available")
            
            # Create point from coordinates
            point = SECP256k1.generator
            
            # Convert to generator point if needed
            if (x, y) != (SECP256k1.generator.x(), SECP256k1.generator.y()):
                # Create custom point (this is complex with ecdsa library)
                # For now, only support generator point multiplication
                if (x, y) != (SECP256k1.generator.x(), SECP256k1.generator.y()):
                    raise NotImplementedError("Custom point multiplication not implemented")
            
            # Multiply by scalar
            result_point = scalar * point
            
            # Return coordinates or None for point at infinity
            if result_point == SECP256k1.generator * 0:  # Point at infinity
                return None
            
            return (result_point.x(), result_point.y())
            
        except Exception as e:
            self.logger.error(f"Error in point multiplication: {e}")
            return None
    
    def is_valid_public_key(self, public_key_hex: str) -> bool:
        """
        Validate public key format and curve membership
        """
        try:
            if not ECDSA_AVAILABLE:
                raise RuntimeError("ecdsa library not available")
            
            if len(public_key_hex) not in [64, 66, 128, 130]:  # Raw/compressed/uncompressed
                return False

            public_key_bytes = bytes.fromhex(public_key_hex)

            self._verifying_key_from_bytes(public_key_bytes)

            return True
            
        except Exception as e:
            self.logger.debug(f"Invalid public key: {e}")
            return False
    
    def is_valid_private_key(self, private_key: int) -> bool:
        """
        Validate private key range
        """
        try:
            # SECP256k1 order
            n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            
            # Private key must be in range [1, n-1]
            return 1 <= private_key < n
            
        except Exception as e:
            self.logger.error(f"Error validating private key: {e}")
            return False
    
    def generate_key_pair(self) -> Tuple[int, str]:
        """
        Generate random key pair using ecdsa library
        Returns (private_key, public_key_hex)
        """
        try:
            if not ECDSA_AVAILABLE:
                raise RuntimeError("ecdsa library not available")
            
            # Generate random signing key
            signing_key = SigningKey.generate(curve=SECP256k1)
            
            # Get private key bytes
            private_key_bytes = signing_key.to_string()
            private_key = int.from_bytes(private_key_bytes, byteorder='big')
            
            # Get public key
            public_key_hex = self.private_key_to_public_key(private_key)
            
            self.logger.debug(f"Generated key pair: private={private_key}, public={public_key_hex}")
            return (private_key, public_key_hex)
            
        except Exception as e:
            self.logger.error(f"Error generating key pair: {e}")
            raise
    
    def assess_key_vulnerability(self, private_key: int) -> float:
        """
        Assess private key vulnerability based on various factors
        Returns vulnerability score between 0.0 (secure) and 1.0 (vulnerable)
        """
        try:
            vulnerability_score = 0.0
            
            # Validate key first
            if not self.is_valid_private_key(private_key):
                return 1.0  # Invalid keys are maximally vulnerable
            
            # Check for weak keys
            if private_key < 1000:  # Very small keys
                vulnerability_score += 0.9
                self.logger.warning(f"Very small private key detected: {private_key}")
            
            # Check for patterns in hex representation
            key_hex = hex(private_key)[2:].lower()
            
            # Check for leading zeros
            if key_hex.startswith('0'*10):
                vulnerability_score += 0.8
                self.logger.warning(f"Private key with many leading zeros: {key_hex[:20]}...")
            
            # Check for leading f's
            if key_hex.startswith('f'*10):
                vulnerability_score += 0.8
                self.logger.warning(f"Private key with many leading f's: {key_hex[:20]}...")
            
            # Check for low entropy (repeating patterns)
            unique_chars = len(set(key_hex))
            if unique_chars < 5:  # Very low entropy
                vulnerability_score += 0.7
                self.logger.warning(f"Private key with low entropy: {unique_chars} unique characters")
            
            # Check mathematical weaknesses
            if self._is_mathematical_weakness(private_key):
                vulnerability_score += 0.6
                self.logger.warning(f"Private key with mathematical weakness: {private_key}")
            
            # Normalize score
            final_score = min(vulnerability_score, 1.0)
            
            if final_score > 0.5:
                self.logger.warning(f"Vulnerable private key detected (score: {final_score:.2f})")
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error assessing key vulnerability: {e}")
            return 0.0
    
    def _is_mathematical_weakness(self, private_key: int) -> bool:
        """
        Check for mathematical weaknesses in private key
        """
        try:
            # SECP256k1 order
            n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
            
            # Check if key is close to curve order
            if abs(private_key - n) < 1000:
                return True
            
            # Check if key is a power of 2
            if private_key > 0 and (private_key & (private_key - 1)) == 0:
                return True
            
            # Check if key is a factorial (up to reasonable limit)
            for i in range(2, 20):
                if math.factorial(i) == private_key:
                    return True
            
            # Check if key is a simple arithmetic sequence
            key_hex = hex(private_key)[2:]
            if len(key_hex) >= 6:
                # Check for sequences like 123456, 111111, etc.
                if key_hex.isdigit() or key_hex == '1' * len(key_hex) or key_hex == '0' * len(key_hex):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking mathematical weakness: {e}")
            return False
    
    def _base58_encode_fallback(self, data: bytes) -> str:
        """
        Fallback base58 encoding implementation
        """
        alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        
        # Convert to integer
        n = int.from_bytes(data, byteorder='big')
        
        # Convert to base58
        encoded = ''
        while n > 0:
            n, r = divmod(n, 58)
            encoded = alphabet[r] + encoded
        
        # Add leading '1's for each leading zero byte
        for byte in data:
            if byte == 0:
                encoded = '1' + encoded
            else:
                break
        
        return encoded
    
    def _base58_decode_fallback(self, encoded: str) -> bytes:
        """
        Fallback base58 decoding implementation
        """
        alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        
        # Convert from base58
        n = 0
        for char in encoded:
            n = n * 58 + alphabet.index(char)
        
        # Convert to bytes
        data = n.to_bytes((n.bit_length() + 7) // 8, byteorder='big')
        
        # Add leading zero bytes for each leading '1'
        for char in encoded:
            if char == '1':
                data = b'\x00' + data
            else:
                break
        
        return data
    
    def get_attack_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about current attack session
        """
        return {
            'attack_id': self.attack_id,
            'libraries_available': {
                'ecdsa': ECDSA_AVAILABLE,
                'cryptography': CRYPTOGRAPHY_AVAILABLE,
                'base58': BASE58_AVAILABLE
            },
            'timestamp': time.time(),
            'logger_name': self.logger.name
        }
    def _verifying_key_from_bytes(self, public_key_bytes: bytes) -> VerifyingKey:
        """Construct a verifying key from raw, compressed, or uncompressed bytes."""
        x, y = self._decode_public_key_bytes(public_key_bytes)
        point = ellipticcurve.Point(SECP256k1.curve, x, y)
        return VerifyingKey.from_public_point(point, curve=SECP256k1)

