#!/usr/bin/env python3
"""
Advanced Shamir Secret Sharing Recovery
Tries multiple approaches to recover the mnemonic including:
1. Direct interpolation with 2 shares
2. Checking if any share has index 0 (vulnerability)
3. Brute force search for possible third share indices
"""

import hashlib
import hmac
import struct
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
EXP = [0] * 512
LOG = [0] * 256

def init_gf256_tables():
    poly = 1
    for i in range(255):
        EXP[i] = poly
        LOG[poly] = i
        poly <<= 1
        if poly & 0x100:
            poly ^= 0x11B
    for i in range(255, 512):
        EXP[i] = EXP[i - 255]

init_gf256_tables()

def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    if a == 0:
        return 0
    if b == 0:
        raise ZeroDivisionError("Division by zero in GF(256)")
    return EXP[(LOG[a] - LOG[b]) % 255]

# ============================================================================
# BIP-39 UTILITIES
# ============================================================================
mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

def extract_share_index_method1(mnemonic_str):
    """Extract index from last word (lower 4 bits)"""
    words = mnemonic_str.strip().split()
    last_word = words[-1]
    word_index = wordlist.index(last_word)
    return word_index & 0x0F

def extract_share_index_method2(mnemonic_str):
    """Extract index from last word (different encoding)"""
    words = mnemonic_str.strip().split()
    last_word = words[-1]
    word_index = wordlist.index(last_word)
    # Try different bit positions
    return (word_index >> 7) & 0x0F

def mnemonic_to_entropy_no_checksum(mnemonic_str):
    words = mnemonic_str.strip().split()
    bits = 0
    for word in words:
        word_index = wordlist.index(word)
        bits = (bits << 11) | word_index
    
    total_bits = len(words) * 11
    checksum_len = total_bits % 32
    entropy_bits = bits >> checksum_len
    entropy_byte_len = (total_bits - checksum_len) // 8
    
    return entropy_bits.to_bytes(entropy_byte_len, 'big')

# ============================================================================
# ADDRESS DERIVATION
# ============================================================================
def mnemonic_to_seed(mnemonic_str, passphrase=""):
    return hashlib.pbkdf2_hmac(
        "sha512",
        mnemonic_str.encode('utf-8'),
        BIP39_SALT + passphrase.encode('utf-8'),
        2048,
        dklen=64
    )

def hmac_sha512(key, data):
    return hmac.new(key, data, hashlib.sha512).digest()

def CKD_priv(k_par, c_par, index):
    if index >= 0x80000000:
        data = b'\x00' + k_par + struct.pack('>L', index)
    else:
        sk = ecdsa.SigningKey.from_string(k_par, curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        pubkey_compressed = vk.to_string("compressed")
        data = pubkey_compressed + struct.pack('>L', index)
    
    I = hmac_sha512(c_par, data)
    IL, IR = I[:32], I[32:]
    k_i = (int.from_bytes(IL, 'big') + int.from_bytes(k_par, 'big')) % ecdsa.SECP256k1.order
    return k_i.to_bytes(32, 'big'), IR

def derive_bip84_address(seed):
    I = hmac_sha512(b"Bitcoin seed", seed)
    master_key, master_chain = I[:32], I[32:]
    
    path = [84 + 0x80000000, 0 + 0x80000000, 0 + 0x80000000, 0, 0]
    k, c = master_key, master_chain
    for index in path:
        k, c = CKD_priv(k, c, index)
    
    sk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1)
    vk = sk.verifying_key
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    prefix = 0x02 if (y & 1) == 0 else 0x03
    compressed_pubkey = bytes([prefix]) + x.to_bytes(32, 'big')
    
    sha256_hash = hashlib.sha256(compressed_pubkey).digest()
    pubkey_hash = hashlib.new('ripemd160', sha256_hash).digest()
    return pubkey_hash.hex()

def verify_mnemonic(mnemonic_str, target_hash):
    """Verify if mnemonic produces the target address"""
    try:
        seed = mnemonic_to_seed(mnemonic_str)
        pubkey_hash = derive_bip84_address(seed)
        return pubkey_hash == target_hash
    except:
        return False

# ============================================================================
# RECOVERY METHODS
# ============================================================================
def lagrange_interpolate_2_shares(x1, y1_bytes, x2, y2_bytes):
    """Interpolate secret at x=0 using 2 shares"""
    y1 = list(y1_bytes)
    y2 = list(y2_bytes)
    
    secret_bytes = []
    for i in range(len(y1)):
        # For linear polynomial: f(0) = (y1*x2 - y2*x1) / (x2 - x1)
        # In GF(256): f(0) = (y1*x2 XOR y2*x1) / (x2 XOR x1)
        numerator = gf_mul(y1[i], x2) ^ gf_mul(y2[i], x1)
        denominator = x1 ^ x2
        
        if denominator == 0:
            raise ValueError("Invalid share indices")
        
        secret_byte = gf_div(numerator, denominator)
        secret_bytes.append(secret_byte)
    
    return bytes(secret_bytes)

def try_recovery_with_indices(share1_str, share2_str, x1, x2, target_hash):
    """Try recovery with specific indices"""
    try:
        y1 = mnemonic_to_entropy_no_checksum(share1_str)
        y2 = mnemonic_to_entropy_no_checksum(share2_str)
        
        secret_entropy = lagrange_interpolate_2_shares(x1, y1, x2, y2)
        recovered_mnemonic = mnemo.to_mnemonic(secret_entropy)
        
        if verify_mnemonic(recovered_mnemonic, target_hash):
            return recovered_mnemonic, True
        return None, False
    except Exception as e:
        return None, False

def brute_force_indices(share1_str, share2_str, target_hash):
    """Try all possible index combinations"""
    print("[*] Brute forcing share indices (0-255)...")
    
    for x1 in range(1, 256):
        for x2 in range(1, 256):
            if x1 == x2:
                continue
            
            if (x1 + x2) % 100 == 0:
                print(f"    Trying indices: x1={x1}, x2={x2}...", end='\r')
            
            mnemonic, success = try_recovery_with_indices(
                share1_str, share2_str, x1, x2, target_hash
            )
            
            if success:
                print(f"\n[+] FOUND! Indices: x1={x1}, x2={x2}")
                return mnemonic, x1, x2
    
    print("\n[-] No valid indices found")
    return None, None, None

def check_direct_share_vulnerability(share1_str, share2_str, target_hash):
    """Check if either share directly contains the secret (index 0 vulnerability)"""
    print("[*] Checking for index-0 vulnerability...")
    
    for i, share in enumerate([share1_str, share2_str], 1):
        try:
            entropy = mnemonic_to_entropy_no_checksum(share)
            mnemonic = mnemo.to_mnemonic(entropy)
            
            if verify_mnemonic(mnemonic, target_hash):
                print(f"[+] Share {i} contains the secret directly!")
                return mnemonic
        except:
            pass
    
    return None

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    SHARE_1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
    SHARE_2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"
    
    print("=" * 70)
    print("ADVANCED SHAMIR SECRET SHARING RECOVERY")
    print("=" * 70)
    print()
    
    # Method 1: Check for direct vulnerability
    result = check_direct_share_vulnerability(SHARE_1, SHARE_2, TARGET_PUBKEY_HASH_HEX)
    if result:
        print(f"\n[✓] RECOVERED MNEMONIC: {result}")
        print("=" * 70)
        exit(0)
    
    # Method 2: Try standard index extraction
    print("\n[*] Method 2: Standard index extraction...")
    x1 = extract_share_index_method1(SHARE_1)
    x2 = extract_share_index_method1(SHARE_2)
    print(f"    Share 1 index: {x1}")
    print(f"    Share 2 index: {x2}")
    
    mnemonic, success = try_recovery_with_indices(
        SHARE_1, SHARE_2, x1, x2, TARGET_PUBKEY_HASH_HEX
    )
    
    if success:
        print(f"\n[✓] RECOVERED MNEMONIC: {mnemonic}")
        print("=" * 70)
        exit(0)
    
    # Method 3: Brute force (limited range for speed)
    print("\n[*] Method 3: Brute force search (limited to 1-16)...")
    for x1 in range(1, 17):
        for x2 in range(1, 17):
            if x1 == x2:
                continue
            
            mnemonic, success = try_recovery_with_indices(
                SHARE_1, SHARE_2, x1, x2, TARGET_PUBKEY_HASH_HEX
            )
            
            if success:
                print(f"[+] FOUND with indices x1={x1}, x2={x2}")
                print(f"\n[✓] RECOVERED MNEMONIC: {mnemonic}")
                print("=" * 70)
                exit(0)
    
    print("\n[-] Recovery failed with all methods")
    print("=" * 70)
