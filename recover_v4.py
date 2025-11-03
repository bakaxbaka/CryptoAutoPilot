#!/usr/bin/env python3
"""
Shamir Secret Sharing Vulnerability Exploit - Version 4
Fixed GF(256) implementation matching pybtc
"""

import hashlib
import hmac
import struct
from mnemonic import Mnemonic
import ecdsa

# Constants
TARGET_PUBKEY_HASH_HEX = "17f33b1f8ef28ac93e4b53753e3817d56a95750e"
BIP39_SALT = b'mnemonic'
mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

print("=" * 70)
print("Shamir Secret Sharing Vulnerability Exploit v4")
print("Fixed GF(256) Implementation")
print("=" * 70)

# --- GF(256) setup - CORRECT VERSION FROM PYBTC ---
def _precompute_gf256_exp_log():
    exp = [0 for i in range(255)]
    log = [0 for i in range(256)]
    poly = 1
    for i in range(255):
        exp[i] = poly
        log[poly] = i
        # THIS IS THE KEY DIFFERENCE: multiply by (x+1), not just x
        poly = (poly << 1) ^ poly  # NOT just poly <<= 1
        if poly & 0x100:
            poly ^= 0x11B
    return exp, log

EXP_TABLE, LOG_TABLE = _precompute_gf256_exp_log()

def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[(LOG_TABLE[a] + LOG_TABLE[b]) % 255]

def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError()
    if a == 0:
        return 0
    # Division: a / b = a * b^-1
    b_inv = EXP_TABLE[(-LOG_TABLE[b]) % 255]
    return gf_mul(a, b_inv)

def gf_add(a, b):
    return a ^ b

def gf_sub(a, b):
    return a ^ b

def gf_pow(a, b):
    if b == 0:
        return 1
    if a == 0:
        return 0
    c = a
    for i in range(b - 1):
        c = gf_mul(c, a)
    return c

# --- Extract share info ---
def extract_share_info(mnemonic_str):
    words = mnemonic_str.strip().split()
    bits = 0
    for word in words:
        bits = (bits << 11) | wordlist.index(word)
    entropy_int = bits >> 4
    share_index = bits & 0x0F
    entropy_bytes = entropy_int.to_bytes(16, 'big')
    return share_index, entropy_bytes

# --- Lagrange interpolation matching pybtc ---
def interpolation(points, x=0):
    """
    Lagrange interpolation in GF(256) - matches pybtc implementation
    points: list of (x, y) tuples
    x: point to evaluate at (default 0 to get secret)
    """
    k = len(points)
    if k < 2:
        raise Exception("Minimum 2 points required")

    p_x = 0
    for j in range(k):
        # Compute Lagrange basis polynomial L_j(x)
        p_j_x = 1
        for m in range(k):
            if m == j:
                continue
            a = gf_sub(x, points[m][0])
            b = gf_sub(points[j][0], points[m][0])
            c = gf_div(a, b)
            p_j_x = gf_mul(p_j_x, c)

        p_j_x = gf_mul(points[j][1], p_j_x)
        p_x = gf_add(p_x, p_j_x)

    return p_x

# --- BIP derivation ---
def hmac_sha512(key, data):
    return hmac.new(key, data, hashlib.sha512).digest()

def CKD_priv(k_par, c_par, index):
    if index >= 0x80000000:
        data = b'\x00' + k_par + struct.pack('>L', index)
    else:
        pubkey = ecdsa.SigningKey.from_string(k_par, curve=ecdsa.SECP256k1).verifying_key
        data = pubkey.to_string("compressed") + struct.pack('>L', index)
    I = hmac_sha512(c_par, data)
    IL, IR = I[:32], I[32:]
    k_i = (int.from_bytes(IL, 'big') + int.from_bytes(k_par, 'big')) % ecdsa.SECP256k1.order
    return k_i.to_bytes(32, 'big'), IR

def derive_bip84_pubkey_hash(seed):
    I = hmac_sha512(b"Bitcoin seed", seed)
    k, c = I[:32], I[32:]
    for index in [84 + 0x80000000, 0 + 0x80000000, 0 + 0x80000000, 0, 0]:
        k, c = CKD_priv(k, c, index)
    pubkey = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
    x, y = pubkey.pubkey.point.x(), pubkey.pubkey.point.y()
    compressed_pubkey = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
    return hashlib.new('ripemd160', hashlib.sha256(compressed_pubkey).digest()).hexdigest()

def mnemonic_to_seed(mnemonic_str):
    return hashlib.pbkdf2_hmac("sha512", mnemonic_str.encode(), BIP39_SALT + b"", 2048, dklen=64)

# --- Main recovery ---
share1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
share2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"

print("\n[1] Extracting share information...")
x1, entropy1 = extract_share_info(share1)
x2, entropy2 = extract_share_info(share2)

print(f"    Share 1: index={x1}, entropy={entropy1.hex()}")
print(f"    Share 2: index={x2}, entropy={entropy2.hex()}")

print("\n[2] Performing Lagrange interpolation in GF(256)...")
print("    Interpolating each byte independently...")

y1_list = list(entropy1)
y2_list = list(entropy2)

secret_bytes = []
for i in range(16):
    points = [(x1, y1_list[i]), (x2, y2_list[i])]
    secret_byte = interpolation(points, x=0)
    secret_bytes.append(secret_byte)

secret_bytes = bytes(secret_bytes)
print(f"    Recovered entropy: {secret_bytes.hex()}")

print("\n[3] Converting to BIP-39 mnemonic...")
recovered_mnemonic = mnemo.to_mnemonic(secret_bytes)
print(f"    Recovered: {recovered_mnemonic}")

print("\n[4] Deriving BIP-84 address (m/84'/0'/0'/0/0)...")
seed = mnemonic_to_seed(recovered_mnemonic)
pubkey_hash = derive_bip84_pubkey_hash(seed)

print(f"    Derived:  {pubkey_hash}")
print(f"    Target:   {TARGET_PUBKEY_HASH_HEX}")

print("\n[5] Verification...")
if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
    print("    ✓ SUCCESS!")
    print("\n" + "=" * 70)
    print("RECOVERED ORIGINAL MNEMONIC:")
    print("=" * 70)
    print(recovered_mnemonic)
    print("=" * 70)
    print("\nThe 1 BTC is available at path: m/84'/0'/0'/0/0")
else:
    print("    ✗ FAILED - mismatch")
