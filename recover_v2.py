#!/usr/bin/env python3
"""
Shamir Secret Sharing Vulnerability Exploit - Version 2
Properly extract entropy and perform Lagrange interpolation
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
print("Shamir Secret Sharing Vulnerability Exploit v2")
print("=" * 70)

# --- GF(256) Galois Field arithmetic ---
EXP, LOG = [0] * 512, [0] * 256
poly = 1
for i in range(255):
    EXP[i] = poly
    LOG[poly] = i
    poly <<= 1
    if poly & 0x100:
        poly ^= 0x11B
for i in range(255, 512):
    EXP[i] = EXP[i - 255]

def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]

def gf_add(a, b):
    return a ^ b

# --- Extract share information ---
def extract_share_info(mnemonic_str):
    """Extract share index and entropy from mnemonic"""
    words = mnemonic_str.strip().split()

    # Convert words to bits
    bits = 0
    for word in words:
        bits = (bits << 11) | wordlist.index(word)

    # For 12 words: 132 bits = 128 entropy + 4 checksum/index
    entropy_int = bits >> 4
    share_index = bits & 0x0F

    entropy_bytes = entropy_int.to_bytes(16, 'big')

    return share_index, entropy_bytes

# --- Lagrange interpolation at x=0 ---
def interpolate_at_zero(x1, y1, x2, y2):
    """
    Lagrange interpolation at x=0 for linear polynomial
    f(0) = y1 * (0-x2)/(x1-x2) + y2 * (0-x1)/(x2-x1)
         = y1 * x2/(x1+x2) + y2 * x1/(x1+x2)   [in GF(256), subtraction is XOR]
         = (y1*x2 + y2*x1) / (x1+x2)
    """
    numerator = gf_add(gf_mul(y1, x2), gf_mul(y2, x1))
    denominator = gf_add(x1, x2)
    return gf_div(numerator, denominator)

# --- BIP derivation functions ---
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
print("    Interpolating each byte at x=0...")

y1_list = list(entropy1)
y2_list = list(entropy2)

secret_bytes = bytes([interpolate_at_zero(x1, y1_list[i], x2, y2_list[i]) for i in range(16)])

print(f"    Secret entropy: {secret_bytes.hex()}")

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
else:
    print("    ✗ FAILED - mismatch")
