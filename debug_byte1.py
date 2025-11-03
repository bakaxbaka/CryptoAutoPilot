#!/usr/bin/env python3
"""Debug byte 1 recovery"""

# GF(256) setup
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

x1, y1 = 3, 0x45  # byte 1 from share 1
x2, y2 = 15, 0x6b  # byte 1 from share 2

print(f"Byte 1 values:")
print(f"  Share 1 (x={x1}): y=0x{y1:02x} ({y1})")
print(f"  Share 2 (x={x2}): y=0x{y2:02x} ({y2})")
print()

matches = []
for test_secret in range(256):
    y1_minus_secret = gf_sub(y1, test_secret)
    slope = gf_div(y1_minus_secret, x1)
    expected_y2 = gf_add(test_secret, gf_mul(slope, x2))

    if expected_y2 == y2:
        matches.append((test_secret, slope))
        print(f"Match: secret=0x{test_secret:02x}, slope=0x{slope:02x}")

if not matches:
    print("No matches found!")
    print("\nLet me check if my GF arithmetic is working...")
    print("Testing: if secret=0x45, slope=0x00, do we get correct values?")
    test_secret = 0x45
    test_slope = 0x00
    test_y1 = gf_add(test_secret, gf_mul(test_slope, x1))
    test_y2 = gf_add(test_secret, gf_mul(test_slope, x2))
    print(f"  f({x1}) = 0x{test_y1:02x}, expected 0x{y1:02x}, match: {test_y1 == y1}")
    print(f"  f({x2}) = 0x{test_y2:02x}, expected 0x{y2:02x}, match: {test_y2 == y2}")

    print("\nMaybe the shares use a different polynomial? Let's check all slopes:")
    for test_slope in range(256):
        test_y1 = gf_add(test_secret, gf_mul(test_slope, x1))
        test_y2 = gf_add(test_secret, gf_mul(test_slope, x2))
        if test_y1 == y1:
            print(f"  If slope=0x{test_slope:02x}: f({x1})=0x{test_y1:02x}✓, f({x2})=0x{test_y2:02x} {'✓' if test_y2==y2 else '✗'}")
