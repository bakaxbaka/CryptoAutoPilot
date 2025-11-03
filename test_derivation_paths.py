#!/usr/bin/env python3
"""
Test different derivation paths to see if we can match the target
"""

import hashlib, hmac, struct
from mnemonic import Mnemonic
import ecdsa

TARGET = '17f33b1f8ef28ac93e4b53753e3817d56a95750e'
mnemo = Mnemonic('english')
wordlist = mnemo.wordlist

SHARE_1 = 'session cigar grape merry useful churn fatal thought very any arm unaware'
SHARE_2 = 'clock fresh security field caution effort gorilla speed plastic common tomato echo'

class GF256:
    def __init__(self):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= 0x11B
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]
    
    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else self.exp[self.log[a] + self.log[b]]
    
    def div(self, a, b):
        return 0 if a == 0 else self.exp[(self.log[a] - self.log[b]) % 255]

gf = GF256()

def get_entropy(s):
    words = s.split()
    bits = 0
    for w in words:
        bits = (bits << 11) | wordlist.index(w)
    return list((bits >> 4).to_bytes(16, 'big'))

def CKD_priv(k, c, idx):
    if idx >= 0x80000000:
        data = b'\x00' + k + struct.pack('>L', idx)
    else:
        sk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1)
        data = sk.verifying_key.to_string('compressed') + struct.pack('>L', idx)
    I = hmac.new(c, data, hashlib.sha512).digest()
    k_new = ((int.from_bytes(I[:32], 'big') + int.from_bytes(k, 'big')) % ecdsa.SECP256k1.order).to_bytes(32, 'big')
    return k_new, I[32:]

def derive_address(mnemonic_str, path_indices):
    """Derive address for given path"""
    try:
        seed = hashlib.pbkdf2_hmac('sha512', mnemonic_str.encode(), b'mnemonic', 2048, 64)
        k, c = seed[:32], seed[32:]
        
        for idx in path_indices:
            k, c = CKD_priv(k, c, idx)
        
        vk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
        x, y = vk.pubkey.point.x(), vk.pubkey.point.y()
        pub = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
        return hashlib.new('ripemd160', hashlib.sha256(pub).digest()).hexdigest()
    except:
        return None

# Test with default indices
y1 = get_entropy(SHARE_1)
y2 = get_entropy(SHARE_2)
x1, x2 = 3, 15

denom = x1 ^ x2
secret = []
for i in range(16):
    num = gf.mul(y1[i], x2) ^ gf.mul(y2[i], x1)
    secret.append(gf.div(num, denom))

test_mnemonic = mnemo.to_mnemonic(bytes(secret))
print(f'Test mnemonic (x1=3, x2=15):')
print(f'  {test_mnemonic}')
print()

# Test different derivation paths
paths_to_test = [
    ("BIP-44 (Legacy)", [0x8000002C, 0x80000000, 0x80000000, 0, 0]),
    ("BIP-49 (SegWit)", [0x80000031, 0x80000000, 0x80000000, 0, 0]),
    ("BIP-84 (Native SegWit)", [0x80000054, 0x80000000, 0x80000000, 0, 0]),
    ("BIP-84 account 1", [0x80000054, 0x80000000, 0x80000001, 0, 0]),
    ("BIP-84 change", [0x80000054, 0x80000000, 0x80000000, 1, 0]),
    ("BIP-84 index 1", [0x80000054, 0x80000000, 0x80000000, 0, 1]),
    ("Simple m/0", [0]),
    ("Simple m/0/0", [0, 0]),
]

print(f'Target: {TARGET}')
print()
print('Testing different derivation paths:')
print()

for name, path in paths_to_test:
    h = derive_address(test_mnemonic, path)
    match = "✓ MATCH!" if h == TARGET else ""
    print(f'{name:25s} -> {h} {match}')

print()
print('Now testing with different share index combinations...')
print()

# Try a few promising index combinations with different paths
test_combinations = [
    (1, 2), (1, 3), (2, 3), (3, 15), (1, 15), (2, 15),
    (4, 5), (5, 6), (1, 4), (2, 5), (3, 6)
]

for x1, x2 in test_combinations:
    denom = x1 ^ x2
    if denom == 0:
        continue
    
    secret = []
    for i in range(16):
        num = gf.mul(y1[i], x2) ^ gf.mul(y2[i], x1)
        secret.append(gf.div(num, denom))
    
    try:
        m = mnemo.to_mnemonic(bytes(secret))
        
        for name, path in paths_to_test:
            h = derive_address(m, path)
            if h == TARGET:
                print(f'\n{"="*70}')
                print(f'[✓] FOUND!')
                print(f'{"="*70}')
                print(f'Indices: x1={x1}, x2={x2}')
                print(f'Path: {name}')
                print(f'Mnemonic: {m}')
                print(f'{"="*70}')
                exit(0)
    except:
        pass

print('\nNo match found with tested combinations and paths.')
