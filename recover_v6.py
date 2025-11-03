#!/usr/bin/env python3
"""
Shamir Secret Sharing Vulnerability Exploit - Version 6
Exploit flawed coefficient generation that never produces 255
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
print("Shamir Secret Sharing Vulnerability Exploit v6")
print("Exploiting Coefficient Generation Bug (coeffs != 255)")
print("=" * 70)

# --- GF(256) setup ---
def _precompute_gf256_exp_log():
    exp = [0 for i in range(255)]
    log = [0 for i in range(256)]
    poly = 1
    for i in range(255):
        exp[i] = poly
        log[poly] = i
        poly = (poly << 1) ^ poly
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
    return EXP_TABLE[(LOG_TABLE[a] * b) % 255]

def eval_poly(coeffs, x):
    result = 0
    for i, coeff in enumerate(coeffs):
        result = gf_add(result, gf_mul(coeff, gf_pow(x, i)))
    return result

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

# --- Brute force with vulnerability filter ---
def find_valid_polynomials_byte(x1, y1, x2, y2):
    """
    Find polynomials where non-constant coefficients != 255
    (exploit the bug where coeffs never equal 255)
    """
    valid_polys = []

    for a2 in range(256):
        # VULNERABILITY: Coefficients are never 255 (except possibly a0, the secret)
        if a2 == 255:
            continue

        # Solve for a1 and a0
        y1_adj = gf_sub(y1, gf_mul(a2, gf_pow(x1, 2)))
        y2_adj = gf_sub(y2, gf_mul(a2, gf_pow(x2, 2)))

        x_diff = gf_sub(x2, x1)
        y_diff = gf_sub(y2_adj, y1_adj)

        if x_diff == 0:
            continue

        a1 = gf_div(y_diff, x_diff)

        # VULNERABILITY FILTER: a1 should also never be 255
        if a1 == 255:
            continue

        a0 = gf_sub(y1_adj, gf_mul(a1, x1))

        # Verify
        check_y1 = eval_poly([a0, a1, a2], x1)
        check_y2 = eval_poly([a0, a1, a2], x2)

        if check_y1 == y1 and check_y2 == y2:
            valid_polys.append((a0, a1, a2))

    return valid_polys

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

print(f"    Share 1: index={x1}")
print(f"    Share 2: index={x2}")

y1_list = list(entropy1)
y2_list = list(entropy2)

print("\n[2] Analyzing first byte with vulnerability filter...")
first_byte_polys = find_valid_polynomials_byte(x1, y1_list[0], x2, y2_list[0])
print(f"    Found {len(first_byte_polys)} valid polynomials (out of 256 total)")
print(f"    Reduced search space by: {(256 - len(first_byte_polys)) / 256 * 100:.1f}%")

if len(first_byte_polys) <= 10:
    print("\n    Valid (a0, a1, a2) for first byte:")
    for poly in first_byte_polys:
        print(f"      {poly}")

print("\n[3] Brute forcing all combinations...")
print("    Building entropy candidates and testing addresses...")

tested = 0
found = False

# For each valid polynomial in first byte
for a0_0, a1_0, a2_0 in first_byte_polys:
    # Try to build a consistent entropy using this a2 pattern
    for a2_pattern in range(256):
        if a2_pattern == 255:  # Skip invalid coefficient
            continue

        secret_bytes = []
        valid = True

        for i in range(16):
            byte_polys = find_valid_polynomials_byte(x1, y1_list[i], x2, y2_list[i])

            # Find polynomial with matching a2
            matching = [p for p in byte_polys if p[2] == a2_pattern]

            if not matching:
                valid = False
                break

            secret_bytes.append(matching[0][0])

        if not valid:
            continue

        tested += 1
        secret_bytes = bytes(secret_bytes)

        try:
            recovered_mnemonic = mnemo.to_mnemonic(secret_bytes)
            seed = mnemonic_to_seed(recovered_mnemonic)
            pubkey_hash = derive_bip84_pubkey_hash(seed)

            if tested % 10 == 0:
                print(f"    Tested {tested} candidates...", end='\r')

            if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
                print(f"\n    ✓ MATCH FOUND! (tested {tested} candidates)")
                print("\n" + "=" * 70)
                print("RECOVERED ORIGINAL MNEMONIC:")
                print("=" * 70)
                print(recovered_mnemonic)
                print("=" * 70)
                print(f"\nSecret entropy: {secret_bytes.hex()}")
                print(f"Common a2 coefficient: {a2_pattern}")
                found = True
                break
        except Exception as e:
            pass

    if found:
        break

if not found:
    print(f"\n    ✗ No match found after testing {tested} candidates")
