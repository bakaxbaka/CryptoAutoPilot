#!/usr/bin/env python3
"""
Shamir Secret Sharing Vulnerability Exploit - Version 3
Brute force each byte by trying all 256 possible secrets
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
print("Shamir Secret Sharing Vulnerability Exploit v3")
print("Brute Force Recovery")
print("=" * 70)

# --- GF(256) setup ---
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

def gf_sub(a, b):
    return a ^ b

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

# --- Brute force secret byte ---
def recover_secret_byte(x1, y1, x2, y2):
    """
    Brute force: try all 256 possible secrets
    For each secret, compute slope from first share
    Verify it produces correct value for second share
    """
    for test_secret in range(256):
        # From f(x1) = secret + slope * x1, we get:
        # slope = (y1 - secret) / x1
        y1_minus_secret = gf_sub(y1, test_secret)
        if x1 == 0:
            # If x1 is 0, then y1 must equal secret
            if y1 == test_secret:
                # Verify with x2
                slope = gf_div(gf_sub(y2, test_secret), x2) if x2 != 0 else 0
                expected_y2 = gf_add(test_secret, gf_mul(slope, x2))
                if expected_y2 == y2:
                    return test_secret
        else:
            slope = gf_div(y1_minus_secret, x1)

            # Verify with second share: f(x2) = secret + slope * x2
            expected_y2 = gf_add(test_secret, gf_mul(slope, x2))

            if expected_y2 == y2:
                return test_secret

    return None  # Should never happen if inputs are valid

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

print("\n[2] Brute forcing each secret byte...")
print("    Trying all 256 possibilities per byte...")

y1_list = list(entropy1)
y2_list = list(entropy2)

secret_bytes = []
for i in range(16):
    secret_byte = recover_secret_byte(x1, y1_list[i], x2, y2_list[i])
    if secret_byte is None:
        print(f"    ERROR: Could not recover byte {i}")
        exit(1)
    secret_bytes.append(secret_byte)
    if i == 0:
        print(f"    Byte 0: 0x{secret_byte:02x}")

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
