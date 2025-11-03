#!/usr/bin/env python3
"""
Complete Shamir Secret Sharing Recovery
Based on the bitaps implementation vulnerability
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

def gf_add(a, b):
    return a ^ b

# ============================================================================
# BIP-39 UTILITIES
# ============================================================================
mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

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
        if not mnemo.check(mnemonic_str):
            return False
        seed = mnemonic_to_seed(mnemonic_str)
        pubkey_hash = derive_bip84_address(seed)
        return pubkey_hash == target_hash
    except:
        return False

# ============================================================================
# LAGRANGE INTERPOLATION IN GF(256)
# ============================================================================
def lagrange_interpolate_at_zero(shares):
    """
    Perform Lagrange interpolation at x=0 to recover the secret.
    shares: list of (x, y_bytes) tuples
    
    For polynomial f(x), we have:
    f(0) = sum over i of: y_i * product over j!=i of: (0 - x_j) / (x_i - x_j)
    
    Since we're evaluating at 0:
    f(0) = sum over i of: y_i * product over j!=i of: (-x_j) / (x_i - x_j)
    f(0) = sum over i of: y_i * product over j!=i of: x_j / (x_j - x_i)
    
    In GF(256), negation is identity (since -a = a in characteristic 2)
    """
    n = len(shares)
    if n < 2:
        raise ValueError("Need at least 2 shares")
    
    # Get the length of entropy from first share
    entropy_len = len(shares[0][1])
    
    # Initialize result
    secret = [0] * entropy_len
    
    # For each share, compute its contribution
    for i in range(n):
        x_i, y_i = shares[i]
        y_i_list = list(y_i)
        
        # Compute Lagrange basis polynomial L_i(0)
        numerator = 1
        denominator = 1
        
        for j in range(n):
            if i == j:
                continue
            x_j = shares[j][0]
            
            # numerator *= (0 - x_j) = x_j (since -x_j = x_j in GF(256))
            numerator = gf_mul(numerator, x_j)
            
            # denominator *= (x_i - x_j) = (x_i XOR x_j)
            denominator = gf_mul(denominator, gf_add(x_i, x_j))
        
        # Compute L_i(0) = numerator / denominator
        if denominator == 0:
            raise ValueError(f"Invalid shares: denominator is zero for share {i}")
        
        lagrange_coeff = gf_div(numerator, denominator)
        
        # Add y_i * L_i(0) to result (for each byte)
        for byte_idx in range(entropy_len):
            contribution = gf_mul(y_i_list[byte_idx], lagrange_coeff)
            secret[byte_idx] = gf_add(secret[byte_idx], contribution)
    
    return bytes(secret)

# ============================================================================
# MAIN RECOVERY
# ============================================================================
def recover_with_two_shares(share1_str, share2_str, x1, x2, target_hash):
    """Recover secret using two shares"""
    print(f"\n[*] Attempting recovery with indices x1={x1}, x2={x2}")
    
    try:
        # Extract entropy from shares
        y1 = mnemonic_to_entropy_no_checksum(share1_str)
        y2 = mnemonic_to_entropy_no_checksum(share2_str)
        
        print(f"    Share 1 entropy: {y1.hex()}")
        print(f"    Share 2 entropy: {y2.hex()}")
        
        # Perform Lagrange interpolation
        shares = [(x1, y1), (x2, y2)]
        secret_entropy = lagrange_interpolate_at_zero(shares)
        
        print(f"    Recovered entropy: {secret_entropy.hex()}")
        
        # Convert to mnemonic
        recovered_mnemonic = mnemo.to_mnemonic(secret_entropy)
        print(f"    Recovered mnemonic: {recovered_mnemonic}")
        
        # Verify
        if verify_mnemonic(recovered_mnemonic, target_hash):
            return recovered_mnemonic, True
        else:
            seed = mnemonic_to_seed(recovered_mnemonic)
            pubkey_hash = derive_bip84_address(seed)
            print(f"    Derived hash: {pubkey_hash}")
            print(f"    Target hash:  {target_hash}")
            return None, False
            
    except Exception as e:
        print(f"    Error: {e}")
        return None, False

def main():
    SHARE_1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
    SHARE_2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"
    
    print("=" * 70)
    print("SHAMIR SECRET SHARING MNEMONIC RECOVERY")
    print("=" * 70)
    print(f"\nTarget pubkey hash: {TARGET_PUBKEY_HASH_HEX}")
    print(f"Target address: bc1q{TARGET_PUBKEY_HASH_HEX}")
    
    # Extract indices from last word (lower 4 bits)
    words1 = SHARE_1.split()
    words2 = SHARE_2.split()
    x1 = wordlist.index(words1[-1]) & 0x0F
    x2 = wordlist.index(words2[-1]) & 0x0F
    
    print(f"\nShare 1 last word: '{words1[-1]}' -> index {x1}")
    print(f"Share 2 last word: '{words2[-1]}' -> index {x2}")
    
    # Try recovery
    mnemonic, success = recover_with_two_shares(
        SHARE_1, SHARE_2, x1, x2, TARGET_PUBKEY_HASH_HEX
    )
    
    if success:
        print(f"\n{'='*70}")
        print("[✓] SUCCESS! MNEMONIC RECOVERED!")
        print(f"{'='*70}")
        print(f"\n{mnemonic}\n")
        print(f"{'='*70}")
        print("The 1 BTC is waiting at path: m/84'/0'/0'/0/0")
        print(f"{'='*70}")
        return mnemonic
    else:
        print(f"\n{'='*70}")
        print("[✗] Recovery failed with standard method")
        print(f"{'='*70}")
        
        # Try brute force with extended range
        print("\n[*] Trying extended brute force search...")
        for test_x1 in range(1, 256):
            for test_x2 in range(1, 256):
                if test_x1 == test_x2:
                    continue
                
                if (test_x1 * 256 + test_x2) % 5000 == 0:
                    print(f"    Progress: x1={test_x1}, x2={test_x2}...", end='\r')
                
                mnemonic, success = recover_with_two_shares(
                    SHARE_1, SHARE_2, test_x1, test_x2, TARGET_PUBKEY_HASH_HEX
                )
                
                if success:
                    print(f"\n\n[✓] FOUND with x1={test_x1}, x2={test_x2}!")
                    print(f"\n{mnemonic}\n")
                    return mnemonic
        
        print("\n[-] No valid solution found")
        return None

if __name__ == "__main__":
    main()
