#!/usr/bin/env python3
"""
Shamir Secret Sharing Mnemonic Recovery Tool
Exploits the vulnerability in the bitaps implementation to recover the original mnemonic
from two shares using GF(256) Lagrange interpolation.
"""

import hashlib
import hmac
import struct
try:
    from mnemonic import Mnemonic
    import ecdsa
except ImportError:
    print("[!] Missing required packages. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mnemonic", "ecdsa"])
    from mnemonic import Mnemonic
    import ecdsa

# ============================================================================
# CONSTANTS
# ============================================================================
TARGET_PUBKEY_HASH_HEX = "17f33b1f8ef28ac93e4b53753e3817d56a95750e"
BIP39_SALT = b'mnemonic'

# ============================================================================
# GF(256) GALOIS FIELD ARITHMETIC
# ============================================================================
# Initialize lookup tables for GF(256) with polynomial 0x11B (AES polynomial)
EXP = [0] * 512
LOG = [0] * 256

def init_gf256_tables():
    """Initialize exponential and logarithm tables for GF(256)"""
    poly = 1
    for i in range(255):
        EXP[i] = poly
        LOG[poly] = i
        poly <<= 1
        if poly & 0x100:
            poly ^= 0x11B  # AES polynomial
    # Extend EXP table for easier computation
    for i in range(255, 512):
        EXP[i] = EXP[i - 255]

init_gf256_tables()

def gf_add(a, b):
    """Addition in GF(256) is XOR"""
    return a ^ b

def gf_sub(a, b):
    """Subtraction in GF(256) is also XOR"""
    return a ^ b

def gf_mul(a, b):
    """Multiplication in GF(256) using lookup tables"""
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    """Division in GF(256) using lookup tables"""
    if a == 0:
        return 0
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(256)")
    return EXP[(LOG[a] - LOG[b]) % 255]

# ============================================================================
# BIP-39 MNEMONIC UTILITIES
# ============================================================================
mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

def extract_share_index(mnemonic_str):
    """
    Extract the share index from the last word of the mnemonic.
    The index is encoded in the lower 4 bits of the last word's index.
    """
    words = mnemonic_str.strip().split()
    last_word = words[-1]
    word_index = wordlist.index(last_word)
    share_index = word_index & 0x0F
    return share_index

def mnemonic_to_entropy_no_checksum(mnemonic_str):
    """
    Convert mnemonic to entropy bytes, stripping the checksum bits.
    For a 12-word mnemonic: 12 * 11 = 132 bits total, 128 bits entropy + 4 bits checksum
    """
    words = mnemonic_str.strip().split()
    
    # Convert words to bit string
    bits = 0
    for word in words:
        word_index = wordlist.index(word)
        bits = (bits << 11) | word_index
    
    # Calculate checksum length
    total_bits = len(words) * 11
    checksum_len = total_bits % 32
    
    # Remove checksum bits
    entropy_bits = bits >> checksum_len
    entropy_byte_len = (total_bits - checksum_len) // 8
    
    return entropy_bits.to_bytes(entropy_byte_len, 'big')

# ============================================================================
# SHAMIR SECRET SHARING RECOVERY
# ============================================================================
def lagrange_interpolate_at_zero(shares_data):
    """
    Perform Lagrange interpolation at x=0 to recover the secret.
    For 2 shares with a linear polynomial (degree 1), we can directly compute:
    secret = (y1 * x2 - y2 * x1) / (x2 - x1) in GF(256)
    
    Which simplifies to: secret = (y1 * x2 + y2 * x1) / (x1 + x2) since subtraction = addition in GF(256)
    """
    if len(shares_data) < 2:
        raise ValueError("Need at least 2 shares for recovery")
    
    # Extract first two shares
    (x1, y1_bytes), (x2, y2_bytes) = shares_data[0], shares_data[1]
    
    # Convert to lists for byte-wise operations
    y1 = list(y1_bytes)
    y2 = list(y2_bytes)
    
    if len(y1) != len(y2):
        raise ValueError("Share entropy lengths don't match")
    
    # Interpolate each byte independently
    secret_bytes = []
    for i in range(len(y1)):
        # Compute: secret[i] = (y1[i] * x2 + y2[i] * x1) / (x1 + x2)
        numerator = gf_add(gf_mul(y1[i], x2), gf_mul(y2[i], x1))
        denominator = gf_add(x1, x2)
        
        if denominator == 0:
            raise ValueError("Invalid share indices: x1 and x2 cannot be equal")
        
        secret_byte = gf_div(numerator, denominator)
        secret_bytes.append(secret_byte)
    
    return bytes(secret_bytes)

# ============================================================================
# BIP-84 ADDRESS DERIVATION
# ============================================================================
def mnemonic_to_seed(mnemonic_str, passphrase=""):
    """Convert BIP-39 mnemonic to seed using PBKDF2"""
    return hashlib.pbkdf2_hmac(
        "sha512",
        mnemonic_str.encode('utf-8'),
        BIP39_SALT + passphrase.encode('utf-8'),
        2048,
        dklen=64
    )

def hmac_sha512(key, data):
    """HMAC-SHA512 helper"""
    return hmac.new(key, data, hashlib.sha512).digest()

def CKD_priv(k_par, c_par, index):
    """
    Child Key Derivation for private keys (BIP-32)
    """
    if index >= 0x80000000:  # Hardened derivation
        data = b'\x00' + k_par + struct.pack('>L', index)
    else:  # Normal derivation
        # Get public key from private key
        sk = ecdsa.SigningKey.from_string(k_par, curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        pubkey_compressed = vk.to_string("compressed")
        data = pubkey_compressed + struct.pack('>L', index)
    
    I = hmac_sha512(c_par, data)
    IL, IR = I[:32], I[32:]
    
    # Compute child key
    k_i = (int.from_bytes(IL, 'big') + int.from_bytes(k_par, 'big')) % ecdsa.SECP256k1.order
    
    return k_i.to_bytes(32, 'big'), IR

def derive_bip84_address(seed):
    """
    Derive BIP-84 address (Native SegWit) at path m/84'/0'/0'/0/0
    Returns the pubkey hash (for bc1q... address)
    """
    # Master key derivation
    I = hmac_sha512(b"Bitcoin seed", seed)
    master_key, master_chain = I[:32], I[32:]
    
    # Derivation path: m/84'/0'/0'/0/0
    path = [
        84 + 0x80000000,  # 84' (purpose - BIP-84)
        0 + 0x80000000,   # 0'  (coin type - Bitcoin)
        0 + 0x80000000,   # 0'  (account)
        0,                # 0   (change - external)
        0                 # 0   (address index)
    ]
    
    k, c = master_key, master_chain
    for index in path:
        k, c = CKD_priv(k, c, index)
    
    # Get public key
    sk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1)
    vk = sk.verifying_key
    
    # Get compressed public key
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    prefix = 0x02 if (y & 1) == 0 else 0x03
    compressed_pubkey = bytes([prefix]) + x.to_bytes(32, 'big')
    
    # Hash to get pubkey hash (HASH160 = RIPEMD160(SHA256(pubkey)))
    sha256_hash = hashlib.sha256(compressed_pubkey).digest()
    pubkey_hash = hashlib.new('ripemd160', sha256_hash).digest()
    
    return pubkey_hash.hex()

# ============================================================================
# MAIN RECOVERY FUNCTION
# ============================================================================
def recover_mnemonic(share1_str, share2_str, target_hash=None):
    """
    Recover the original mnemonic from two Shamir shares.
    """
    print("=" * 70)
    print("SHAMIR SECRET SHARING MNEMONIC RECOVERY")
    print("=" * 70)
    print()
    
    # Extract share indices
    x1 = extract_share_index(share1_str)
    x2 = extract_share_index(share2_str)
    
    print(f"[*] Share 1 index: {x1}")
    print(f"[*] Share 2 index: {x2}")
    print()
    
    # Extract entropy from shares (without checksum)
    y1 = mnemonic_to_entropy_no_checksum(share1_str)
    y2 = mnemonic_to_entropy_no_checksum(share2_str)
    
    print(f"[*] Share 1 entropy: {y1.hex()}")
    print(f"[*] Share 2 entropy: {y2.hex()}")
    print()
    
    # Perform Lagrange interpolation to recover secret
    print("[*] Performing Lagrange interpolation in GF(256)...")
    shares_data = [(x1, y1), (x2, y2)]
    secret_entropy = lagrange_interpolate_at_zero(shares_data)
    
    print(f"[*] Recovered entropy: {secret_entropy.hex()}")
    print()
    
    # Convert entropy to valid BIP-39 mnemonic (with checksum)
    recovered_mnemonic = mnemo.to_mnemonic(secret_entropy)
    
    print("[+] RECOVERED MNEMONIC:")
    print(f"    {recovered_mnemonic}")
    print()
    
    # Derive address and verify
    print("[*] Deriving BIP-84 address (m/84'/0'/0'/0/0)...")
    seed = mnemonic_to_seed(recovered_mnemonic)
    pubkey_hash = derive_bip84_address(seed)
    
    print(f"[*] Derived pubkey hash: {pubkey_hash}")
    print()
    
    if target_hash:
        if pubkey_hash == target_hash:
            print("[✓] SUCCESS! Pubkey hash matches target!")
            print(f"[✓] Target address: bc1q{target_hash}")
            return recovered_mnemonic, True
        else:
            print("[✗] WARNING: Pubkey hash does NOT match target!")
            print(f"[✗] Expected: {target_hash}")
            print(f"[✗] Got:      {pubkey_hash}")
            return recovered_mnemonic, False
    
    return recovered_mnemonic, None

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # The two shares provided
    SHARE_1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
    SHARE_2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"
    
    # Recover the mnemonic
    recovered, success = recover_mnemonic(SHARE_1, SHARE_2, TARGET_PUBKEY_HASH_HEX)
    
    print("=" * 70)
    if success:
        print("RECOVERY COMPLETE!")
        print(f"Original Mnemonic: {recovered}")
        print()
        print("You can now use this mnemonic to access the Bitcoin wallet.")
        print("The 1 BTC is waiting at derivation path: m/84'/0'/0'/0/0")
    elif success is False:
        print("RECOVERY FAILED - Hash mismatch")
        print("The recovered mnemonic may be incorrect.")
    else:
        print("RECOVERY COMPLETE (no verification)")
        print(f"Recovered Mnemonic: {recovered}")
    print("=" * 70)
