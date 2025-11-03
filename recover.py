#!/usr/bin/env python3
"""
Shamir Secret Sharing Mnemonic Recovery Tool
============================================

This script attempts to recover a BIP-39 mnemonic from two Shamir Secret Sharing shares
by exploiting the vulnerability in the bitaps implementation.

Usage:
    python3 recover.py [--passphrase PASSPHRASE] [--fast]

Options:
    --passphrase    BIP-39 passphrase (if any)
    --fast          Quick test with common indices only
"""

import hashlib, hmac, struct, sys, argparse
from mnemonic import Mnemonic
import ecdsa

# Configuration
TARGET_HASH = '17f33b1f8ef28ac93e4b53753e3817d56a95750e'
SHARE_1 = 'session cigar grape merry useful churn fatal thought very any arm unaware'
SHARE_2 = 'clock fresh security field caution effort gorilla speed plastic common tomato echo'

# Initialize
mnemo = Mnemonic('english')
wordlist = mnemo.wordlist

class GF256:
    """Galois Field GF(2^8) arithmetic for Shamir Secret Sharing"""
    def __init__(self):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x = (x << 1) ^ (0x11B if x & 0x100 else 0)
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]
    
    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else self.exp[self.log[a] + self.log[b]]
    
    def div(self, a, b):
        return 0 if a == 0 else self.exp[(self.log[a] - self.log[b]) % 255]

gf = GF256()

def extract_entropy(share_str):
    """Extract entropy from share mnemonic (without checksum)"""
    words = share_str.split()
    bits = sum(wordlist.index(w) << (11 * (len(words) - 1 - i)) for i, w in enumerate(words))
    return list((bits >> 4).to_bytes(16, 'big'))

def interpolate_secret(y1, y2, x1, x2):
    """Perform Lagrange interpolation to recover secret at x=0"""
    denom = x1 ^ x2
    if denom == 0:
        return None
    return bytes([gf.div(gf.mul(y1[i], x2) ^ gf.mul(y2[i], x1), denom) for i in range(16)])

def derive_key(seed, path):
    """Derive key for given BIP-32 path"""
    k, c = seed[:32], seed[32:]
    for idx in path:
        data = (b'\x00' + k if idx >= 0x80000000 else 
                ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key.to_string('compressed')
               ) + struct.pack('>L', idx)
        I = hmac.new(c, data, hashlib.sha512).digest()
        k = ((int.from_bytes(I[:32], 'big') + int.from_bytes(k, 'big')) % ecdsa.SECP256k1.order).to_bytes(32, 'big')
        c = I[32:]
    return k

def get_address_hash(mnemonic_str, passphrase, path):
    """Derive address hash for given mnemonic and path"""
    try:
        seed = hashlib.pbkdf2_hmac('sha512', mnemonic_str.encode(), b'mnemonic' + passphrase.encode(), 2048, 64)
        k = derive_key(seed, path)
        vk = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
        x, y = vk.pubkey.point.x(), vk.pubkey.point.y()
        pub = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
        return hashlib.new('ripemd160', hashlib.sha256(pub).digest()).hexdigest()
    except:
        return None

def search(y1, y2, passphrase='', fast_mode=False):
    """Search for correct share indices"""
    paths = [
        ('BIP-84 m/84\'/0\'/0\'/0/0', [0x80000054, 0x80000000, 0x80000000, 0, 0]),
        ('BIP-44 m/44\'/0\'/0\'/0/0', [0x8000002C, 0x80000000, 0x80000000, 0, 0]),
        ('BIP-49 m/49\'/0\'/0\'/0/0', [0x80000031, 0x80000000, 0x80000000, 0, 0]),
    ]
    
    # Determine search range
    if fast_mode:
        indices = [(i, j) for i in range(1, 17) for j in range(1, 17) if i != j]
        print(f'[*] Fast mode: testing {len(indices)} combinations')
    else:
        indices = [(i, j) for i in range(1, 256) for j in range(1, 256) if i != j]
        print(f'[*] Full mode: testing {len(indices)} combinations')
    
    print(f'[*] Target: {TARGET_HASH}')
    print(f'[*] Passphrase: {"(none)" if not passphrase else "(provided)"}')
    print()
    
    checked = 0
    start = __import__('time').time()
    
    for x1, x2 in indices:
        checked += 1
        
        if checked % 5000 == 0:
            elapsed = __import__('time').time() - start
            rate = checked / elapsed if elapsed > 0 else 0
            remaining = (len(indices) - checked) / rate if rate > 0 else 0
            print(f'Progress: {checked}/{len(indices)} ({100*checked/len(indices):.1f}%) - '
                  f'{rate:.0f}/sec - ETA: {remaining/60:.1f}min', end='\r')
        
        secret = interpolate_secret(y1, y2, x1, x2)
        if not secret:
            continue
        
        try:
            mnemonic = mnemo.to_mnemonic(secret)
            
            for path_name, path in paths:
                h = get_address_hash(mnemonic, passphrase, path)
                if h == TARGET_HASH:
                    print(f'\n\n{"="*70}')
                    print('[✓] SUCCESS! MNEMONIC RECOVERED!')
                    print(f'{"="*70}')
                    print(f'Share indices: x1={x1}, x2={x2}')
                    print(f'Derivation path: {path_name}')
                    if passphrase:
                        print(f'Passphrase: {passphrase}')
                    print(f'\nRecovered mnemonic:')
                    print(f'  {mnemonic}')
                    print(f'\n{"="*70}')
                    return mnemonic
        except:
            pass
    
    print(f'\n\n[✗] No match found after {checked} attempts')
    return None

def main():
    parser = argparse.ArgumentParser(description='Recover mnemonic from Shamir shares')
    parser.add_argument('--passphrase', default='', help='BIP-39 passphrase')
    parser.add_argument('--fast', action='store_true', help='Fast mode (test indices 1-16 only)')
    args = parser.parse_args()
    
    print('='*70)
    print('SHAMIR SECRET SHARING MNEMONIC RECOVERY')
    print('='*70)
    print()
    print(f'Share 1: {SHARE_1}')
    print(f'Share 2: {SHARE_2}')
    print()
    
    # Extract entropy
    y1 = extract_entropy(SHARE_1)
    y2 = extract_entropy(SHARE_2)
    
    # Search
    result = search(y1, y2, args.passphrase, args.fast)
    
    if not result:
        print()
        print('Possible reasons:')
        print('  1. Incorrect passphrase (try --passphrase option)')
        print('  2. Custom derivation path not tested')
        print('  3. Three shares required (not just two)')
        print('  4. Incorrect target address')
        print()
        print('Try running with --fast first, then full search if needed.')
        sys.exit(1)

if __name__ == '__main__':
    main()
