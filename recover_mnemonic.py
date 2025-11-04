#!/usr/bin/env python3
"""
Shamir Secret Sharing Vulnerability Exploit
Recovers the original BIP-39 mnemonic from 2 shares due to linear polynomial vulnerability
"""

import hashlib
import hmac
import struct

import ecdsa
from mnemonic import Mnemonic

# --- Constants ---
TARGET_PUBKEY_HASH_HEX = "17f33b1f8ef28ac93e4b53753e3817d56a95750e"
BIP39_SALT = b"mnemonic"
mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

print("=" * 70)
print("Shamir Secret Sharing Vulnerability Exploit")
print("=" * 70)

# --- GF(256) math ---
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

def gf_add(a, b):
    return a ^ b

def gf_sub(a, b):
    return a ^ b

def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]

# --- Share processing helpers ---
def extract_share_index(mnemonic: str):
    """Extract the share index from the last word's lower 4 bits."""
    words = mnemonic.strip().split()
    last_word = words[-1]
    return wordlist.index(last_word) & 0x0F

def mnemonic_to_entropy_no_checksum(mnemonic: str) -> bytes:
    """Convert mnemonic to entropy bytes, removing checksum bits."""
    words = mnemonic.strip().split()
    bits = 0
    for word in words:
        bits = (bits << 11) | wordlist.index(word)

    checksum_len = (len(words) * 11) % 32
    entropy_bits = bits >> checksum_len
    entropy_bytes = (len(words) * 11 - checksum_len) // 8
    return entropy_bits.to_bytes(entropy_bytes, "big")

# --- Interpolation ---
def compute_secret_byte(y1, y2, x1, x2):
    """Recover the secret byte using Lagrange interpolation for linear polynomials."""
    numerator = gf_add(gf_mul(y1, x2), gf_mul(y2, x1))
    denominator = gf_add(x1, x2)
    return gf_div(numerator, denominator)

# --- BIP-84 derivation functions ---
def mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
    """Convert mnemonic to BIP-39 seed using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        "sha512",
        mnemonic.encode(),
        BIP39_SALT + passphrase.encode(),
        2048,
        dklen=64,
    )

def hmac_sha512(key, data):
    """HMAC-SHA512 helper."""
    return hmac.new(key, data, hashlib.sha512).digest()

def CKD_priv(k_par, c_par, index):
    """Child Key Derivation for private keys (BIP-32)."""
    if index >= 0x80000000:
        data = b"\x00" + k_par + struct.pack(">L", index)
    else:
        pubkey = ecdsa.SigningKey.from_string(k_par, curve=ecdsa.SECP256k1).verifying_key
        data = pubkey.to_string("compressed") + struct.pack(">L", index)

    I = hmac_sha512(c_par, data)
    IL, IR = I[:32], I[32:]
    k_i = (int.from_bytes(IL, "big") + int.from_bytes(k_par, "big")) % ecdsa.SECP256k1.order
    return k_i.to_bytes(32, "big"), IR

def derive_bip84_pubkey_hash(seed: bytes) -> str:
    """Derive the BIP-84 address pubkey hash (path m/84'/0'/0'/0/0)."""
    I = hmac_sha512(b"Bitcoin seed", seed)
    k, c = I[:32], I[32:]

    derivation_path = [
        84 + 0x80000000,
        0 + 0x80000000,
        0 + 0x80000000,
        0,
        0,
    ]

    for index in derivation_path:
        k, c = CKD_priv(k, c, index)

    pubkey = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
    x, y = pubkey.pubkey.point.x(), pubkey.pubkey.point.y()
    compressed_pubkey = bytes([0x02 + (y & 1)]) + x.to_bytes(32, "big")

    sha256_hash = hashlib.sha256(compressed_pubkey).digest()
    return hashlib.new("ripemd160", sha256_hash).hexdigest()

# --- Main recovery process ---
print("\n[1] Loading shares...")
share1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
share2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"
print(f"    Share 1: {share1}")
print(f"    Share 2: {share2}")

print("\n[2] Extracting share indices...")
x1 = extract_share_index(share1)
x2 = extract_share_index(share2)
print(f"    Share 1 index (x1): {x1}")
print(f"    Share 2 index (x2): {x2}")

print("\n[3] Converting shares to entropy (without checksum)...")
y1_bytes = mnemonic_to_entropy_no_checksum(share1)
y2_bytes = mnemonic_to_entropy_no_checksum(share2)
y1 = list(y1_bytes)
y2 = list(y2_bytes)
print(f"    Share 1 entropy length: {len(y1)} bytes")
print(f"    Share 2 entropy length: {len(y2)} bytes")

print("\n[4] Performing Lagrange interpolation in GF(256)...")
secret_bytes = bytes(compute_secret_byte(y1[i], y2[i], x1, x2) for i in range(len(y1)))
print(f"    Recovered entropy: {secret_bytes.hex()}")

print("\n[5] Converting entropy to valid BIP-39 mnemonic...")
recovered_mnemonic = mnemo.to_mnemonic(secret_bytes)
print(f"    Recovered mnemonic: {recovered_mnemonic}")

print("\n[6] Deriving BIP-84 address from mnemonic...")
seed = mnemonic_to_seed(recovered_mnemonic)
pubkey_hash = derive_bip84_pubkey_hash(seed)
print(f"    Derived pubkey hash: {pubkey_hash}")
print(f"    Target pubkey hash:  {TARGET_PUBKEY_HASH_HEX}")

print("\n[7] Verification...")
if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
    print("    \u2713 SUCCESS! Pubkey hashes match!")
    print("\n" + "=" * 70)
    print("RECOVERED ORIGINAL MNEMONIC:")
    print("=" * 70)
    print(recovered_mnemonic)
    print("=" * 70)
    print(f"\nCorresponding to address: bc1qyj...(pubkey hash: {pubkey_hash})")
    print("Path: m/84'/0'/0'/0/0")
else:
    print("    \u2717 FAILED! Pubkey hashes do not match.")
    print("    The recovered mnemonic does not correspond to the target address.")

print("\n" + "=" * 70)
print("Exploit Complete")
print("=" * 70)
