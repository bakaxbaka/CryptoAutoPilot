#!/usr/bin/env python3
"""
Ultimate recovery script - tries every possible combination
"""

import hashlib, hmac, struct
from mnemonic import Mnemonic
import ecdsa
import sys

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

y1 = get_entropy(SHARE_1)
y2 = get_entropy(SHARE_2)

# All paths to test
paths = [
    ("BIP-44 m/44'/0'/0'/0/0", [0x8000002C, 0x80000000, 0x80000000, 0, 0]),
    ("BIP-49 m/49'/0'/0'/0/0", [0x80000031, 0x80000000, 0x80000000, 0, 0]),
    ("BIP-84 m/84'/0'/0'/0/0", [0x80000054, 0x80000000, 0x80000000, 0, 0]),
]

print(f'Target: {TARGET}')
print(f'Searching through all combinations...')
print()

checked = 0
start = __import__('time').time()

# Search through all index combinations
for x1 in range(1, 256):
    for x2 in range(1, 256):
        if x1 == x2:
            continue
        
        checked += 1
        
        if checked % 10000 == 0:
            elapsed = __import__('time').time() - start
            rate = checked / elapsed if elapsed > 0 else 0
            print(f'Checked: {checked} combinations ({rate:.0f}/sec) - x1={x1}, x2={x2}', end='\r')
        
        denom = x1 ^ x2
        if denom == 0:
            continue
        
        try:
            secret = []
            for i in range(16):
                num = gf.mul(y1[i], x2) ^ gf.mul(y2[i], x1)
                secret.append(gf.div(num, denom))
            
            m = mnemo.to_mnemonic(bytes(secret))
            
            # Test all paths
            for path_name, path in paths:
                h = derive_address(m, path)
                if h == TARGET:
                    print(f'\n\n{"="*70}')
                    print('[✓] SUCCESS! MNEMONIC RECOVERED!')
                    print(f'{"="*70}')
                    print(f'Share indices: x1={x1}, x2={x2}')
                    print(f'Derivation path: {path_name}')
                    print(f'\nRecovered mnemonic:')
                    print(f'  {m}')
                    print(f'\n{"="*70}')
                    sys.exit(0)
        except:
            pass

print(f'\n\nSearched {checked} combinations across all paths. No match found.')
print()
print('This suggests:')
print('1. The target address may have a passphrase')
print('2. The shares may be from a different mnemonic')
print('3. Additional information is needed')
