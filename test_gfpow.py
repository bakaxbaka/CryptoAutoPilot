#!/usr/bin/env python3
"""Test GF(256) power function"""

# My implementation
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

# My gf_pow using logs
def gf_pow_v1(a, b):
    if b == 0:
        return 1
    if a == 0:
        return 0
    return EXP_TABLE[(LOG_TABLE[a] * b) % 255]

# Reference implementation from pybtc
def gf_pow_v2(a, b):
    if b == 0:
        return 1
    if a == 0:
        return 0
    c = a
    for i in range(b - 1):
        c = gf_mul(c, a)
    return c

# Test both
for a in [2, 3, 5, 10, 15]:
    for b in [0, 1, 2, 3, 4]:
        r1 = gf_pow_v1(a, b)
        r2 = gf_pow_v2(a, b)
        match = "✓" if r1 == r2 else "✗"
        print(f"{match} gf_pow({a}, {b}): v1={r1}, v2={r2}")

