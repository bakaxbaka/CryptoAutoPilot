#!/usr/bin/env python3
"""
Final recovery attempt with corrected GF(256) implementation
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

# Proper GF256 setup
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
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]
    
    def div(self, a, b):
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError()
        return self.exp[(self.log[a] - self.log[b]) % 255]

gf = GF256()

def get_entropy(s):
    words = s.split()
    bits = 0
    for w in words:
        bits = (bits << 11) | wordlist.index(w)
    return list((bits >> 4).to_bytes(16, 'big'))

def derive_hash(entropy_bytes):
    try:
        m = mnemo.to_mnemonic(bytes(entropy_bytes))
        if not mnemo.check(m):
            return None
        
        seed = hashlib.pbkdf2_hmac('sha512', m.encode(), b'mnemonic', 2048, 64)
        k, c = seed[:32], seed[32:]
        
        # BIP-84: m/84'/0'/0'/0/0
        for idx in [0x80000054, 0x80000000, 0x80000000, 0, 0]:
            if idx >= 0x80000000:
                data = b'\x00' + k + struct.pack('>L', idx)
            else:
                sk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1)
                data = sk.verifying_key.to_string('compressed') + struct.pack('>L', idx)
            I = hmac.new(c, data, hashlib.sha512).digest()
            k = ((int.from_bytes(I[:32], 'big') + int.from_bytes(k, 'big')) % ecdsa.SECP256k1.order).to_bytes(32, 'big')
            c = I[32:]
        
        vk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
        x, y = vk.pubkey.point.x(), vk.pubkey.point.y()
        pub = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
        return hashlib.new('ripemd160', hashlib.sha256(pub).digest()).hexdigest()
    except:
        return None

y1 = get_entropy(SHARE_1)
y2 = get_entropy(SHARE_2)

x1_default = wordlist.index(SHARE_1.split()[-1]) & 0x0F
x2_default = wordlist.index(SHARE_2.split()[-1]) & 0x0F

print(f'Default indices: x1={x1_default}, x2={x2_default}')
print(f'Target: {TARGET}')
print()
print('Searching for correct indices...')
print()

checked = 0
start_time = __import__('time').time()

# Search with progress indicator
for x1 in range(1, 256):
    for x2 in range(1, 256):
        if x1 == x2:
            continue
        
        checked += 1
        
        # Progress every 5000 checks
        if checked % 5000 == 0:
            elapsed = __import__('time').time() - start_time
            rate = checked / elapsed
            remaining = (256 * 255 - checked) / rate
            print(f'Progress: {checked}/{256*255} ({100*checked/(256*255):.1f}%) - '
                  f'{rate:.0f} checks/sec - ETA: {remaining/60:.1f} min', end='\r')
        
        denom = x1 ^ x2
        if denom == 0:
            continue
        
        try:
            secret = []
            for i in range(16):
                num = gf.mul(y1[i], x2) ^ gf.mul(y2[i], x1)
                secret.append(gf.div(num, denom))
            
            h = derive_hash(secret)
            if h == TARGET:
                m = mnemo.to_mnemonic(bytes(secret))
                print(f'\n\n{"="*70}')
                print('[✓] SUCCESS! MNEMONIC RECOVERED!')
                print(f'{"="*70}')
                print(f'Share indices: x1={x1}, x2={x2}')
                print(f'\nRecovered mnemonic:')
                print(f'  {m}')
                print(f'\n{"="*70}')
                print('The 1 BTC is waiting at path: m/84\'/0\'/0\'/0/0')
                print(f'{"="*70}')
                sys.exit(0)
        except:
            pass

print(f'\n\nSearched {checked} combinations. No match found.')
