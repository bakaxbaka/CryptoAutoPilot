#!/usr/bin/env python3
"""
Fast brute force search for correct share indices
"""

import hashlib, hmac, struct
from mnemonic import Mnemonic
import ecdsa

TARGET = '17f33b1f8ef28ac93e4b53753e3817d56a95750e'
mnemo = Mnemonic('english')
wordlist = mnemo.wordlist

SHARE_1 = 'session cigar grape merry useful churn fatal thought very any arm unaware'
SHARE_2 = 'clock fresh security field caution effort gorilla speed plastic common tomato echo'

def get_entropy(s):
    words = s.split()
    bits = 0
    for w in words:
        bits = (bits << 11) | wordlist.index(w)
    return (bits >> 4).to_bytes(16, 'big')

y1 = list(get_entropy(SHARE_1))
y2 = list(get_entropy(SHARE_2))

# GF256 tables
EXP = [0] * 512
LOG = [0] * 256
p = 1
for i in range(255):
    EXP[i] = p
    if p in LOG:
        pass  # Skip duplicates
    else:
        LOG[p] = i
    p <<= 1
    if p & 0x100:
        p ^= 0x11B
for i in range(255, 512):
    EXP[i] = EXP[i - 255]

def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    if a == 0:
        return 0
    if b == 0:
        return 0
    return EXP[(LOG[a] - LOG[b]) % 255]

def derive_hash(entropy_bytes):
    try:
        m = mnemo.to_mnemonic(bytes(entropy_bytes))
        seed = hashlib.pbkdf2_hmac('sha512', m.encode(), b'mnemonic', 2048, 64)
        k, c = seed[:32], seed[32:]
        
        # BIP-84 path: m/84'/0'/0'/0/0
        for idx in [0x80000054, 0x80000000, 0x80000000, 0, 0]:
            if idx >= 0x80000000:
                data = b'\x00' + k + struct.pack('>L', idx)
            else:
                sk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1)
                data = sk.verifying_key.to_string('compressed') + struct.pack('>L', idx)
            I = hmac.new(c, data, hashlib.sha512).digest()
            k_int = (int.from_bytes(I[:32], 'big') + int.from_bytes(k, 'big')) % ecdsa.SECP256k1.order
            k = k_int.to_bytes(32, 'big')
            c = I[32:]
        
        sk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        pub = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
        return hashlib.new('ripemd160', hashlib.sha256(pub).digest()).hexdigest()
    except Exception as e:
        return None

print('Starting brute force search...')
print(f'Target: {TARGET}')
print()

checked = 0
for x1 in range(1, 256):
    for x2 in range(1, 256):
        if x1 == x2:
            continue
        
        checked += 1
        if checked % 1000 == 0:
            print(f'Checked {checked} combinations... (x1={x1}, x2={x2})', end='\r')
        
        # Compute secret using Lagrange interpolation
        denominator = x1 ^ x2
        if denominator == 0:
            continue
        
        secret = []
        for i in range(16):
            numerator = gf_mul(y1[i], x2) ^ gf_mul(y2[i], x1)
            secret_byte = gf_div(numerator, denominator)
            secret.append(secret_byte)
        
        h = derive_hash(secret)
        if h == TARGET:
            mnemonic = mnemo.to_mnemonic(bytes(secret))
            print(f'\n\n{"="*70}')
            print(f'[✓] FOUND!')
            print(f'{"="*70}')
            print(f'Share indices: x1={x1}, x2={x2}')
            print(f'Recovered mnemonic:')
            print(f'  {mnemonic}')
            print(f'{"="*70}')
            exit(0)

print(f'\n\nSearched {checked} combinations. No match found.')
