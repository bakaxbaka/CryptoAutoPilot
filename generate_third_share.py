#!/usr/bin/env python3
"""
Generate and test possible third shares
Since we have a 3-of-5 threshold scheme, we can try to generate what the third share might be
and use all 3 shares to recover the secret
"""

import hashlib, hmac, struct
from mnemonic import Mnemonic
import ecdsa

TARGET = '17f33b1f8ef28ac93e4b53753e3817d56a95750e'
mnemo = Mnemonic('english')
wordlist = mnemo.wordlist

SHARE_1 = 'session cigar grape merry useful churn fatal thought very any arm unaware'
SHARE_2 = 'clock fresh security field caution effort gorilla speed plastic common tomato echo'

# GF256 setup
EXP = [0] * 512
LOG = [0] * 256
p = 1
for i in range(255):
    EXP[i] = p
    LOG[p] = i
    p = (p << 1) ^ (0x11B if p & 0x100 else 0)
for i in range(255, 512):
    EXP[i] = EXP[i - 255]

def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]

def gf_div(a, b):
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]

def get_entropy(s):
    words = s.split()
    bits = 0
    for w in words:
        bits = (bits << 11) | wordlist.index(w)
    return list((bits >> 4).to_bytes(16, 'big'))

def derive_hash(entropy_bytes):
    try:
        m = mnemo.to_mnemonic(bytes(entropy_bytes))
        seed = hashlib.pbkdf2_hmac('sha512', m.encode(), b'mnemonic', 2048, 64)
        k, c = seed[:32], seed[32:]
        
        for idx in [0x80000054, 0x80000000, 0x80000000, 0, 0]:
            data = (b'\x00' + k if idx >= 0x80000000 else 
                   ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key.to_string('compressed')) + struct.pack('>L', idx)
            I = hmac.new(c, data, hashlib.sha512).digest()
            k = ((int.from_bytes(I[:32], 'big') + int.from_bytes(k, 'big')) % ecdsa.SECP256k1.order).to_bytes(32, 'big')
            c = I[32:]
        
        vk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
        x, y = vk.pubkey.point.x(), vk.pubkey.point.y()
        pub = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
        return hashlib.new('ripemd160', hashlib.sha256(pub).digest()).hexdigest()
    except:
        return None

def lagrange_3_shares(shares):
    """Lagrange interpolation with 3 shares at x=0"""
    secret = [0] * 16
    
    for i, (x_i, y_i) in enumerate(shares):
        # Compute Lagrange basis L_i(0)
        num, den = 1, 1
        for j, (x_j, _) in enumerate(shares):
            if i != j:
                num = gf_mul(num, x_j)
                den = gf_mul(den, x_i ^ x_j)
        
        if den == 0:
            return None
        
        coeff = gf_div(num, den)
        for byte_idx in range(16):
            secret[byte_idx] ^= gf_mul(y_i[byte_idx], coeff)
    
    return secret

y1 = get_entropy(SHARE_1)
y2 = get_entropy(SHARE_2)

# Extract indices from last words
x1 = wordlist.index(SHARE_1.split()[-1]) & 0x0F
x2 = wordlist.index(SHARE_2.split()[-1]) & 0x0F

print(f'Share 1 index: {x1}')
print(f'Share 2 index: {x2}')
print(f'Target: {TARGET}')
print()
print('Testing with possible third share indices...')
print()

# Try different third share indices
for x3 in range(1, 16):
    if x3 == x1 or x3 == x2:
        continue
    
    print(f'Testing x3={x3}...', end=' ')
    
    # For each possible third share, we need to guess its entropy
    # But we can use the fact that with 3 shares and a degree-2 polynomial,
    # we can solve for the secret
    
    # Actually, let's try a different approach:
    # If the polynomial is degree 1 (vulnerability), then 2 shares are enough
    # Let's try different interpretations of the indices
    
    for test_x1 in [x1, x1+16, x1+32]:
        for test_x2 in [x2, x2+16, x2+32]:
            if test_x1 == test_x2:
                continue
            
            denom = test_x1 ^ test_x2
            if denom == 0:
                continue
            
            secret = []
            for i in range(16):
                num = gf_mul(y1[i], test_x2) ^ gf_mul(y2[i], test_x1)
                secret.append(gf_div(num, denom))
            
            h = derive_hash(secret)
            if h == TARGET:
                m = mnemo.to_mnemonic(bytes(secret))
                print(f'\n\n{"="*70}')
                print('[✓] FOUND!')
                print(f'{"="*70}')
                print(f'Indices: x1={test_x1}, x2={test_x2}')
                print(f'Mnemonic: {m}')
                print(f'{"="*70}')
                exit(0)
    
    print('no match')

print('\nNo solution found with this approach.')
