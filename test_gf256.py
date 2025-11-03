#!/usr/bin/env python3
"""
Test GF(256) arithmetic and Lagrange interpolation
"""

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

print("Testing GF(256) arithmetic:")
print(f"gf_add(3, 5) = {gf_add(3, 5)}")
print(f"gf_mul(3, 5) = {gf_mul(3, 5)}")
print(f"gf_div(15, 3) = {gf_div(15, 3)}")

# Test with a known linear polynomial: f(x) = 100 + 50*x
# f(3) should equal 100 ^ (50 * 3)
# f(15) should equal 100 ^ (50 * 15)

secret = 100
slope = 50

y3 = gf_add(secret, gf_mul(slope, 3))
y15 = gf_add(secret, gf_mul(slope, 15))

print(f"\nTest polynomial: f(x) = {secret} + {slope}*x")
print(f"f(3) = {y3}")
print(f"f(15) = {y15}")

# Now try to recover secret from (3, y3) and (15, y15)
# Using Lagrange: f(0) = y1*(0-x2)/(x1-x2) + y2*(0-x1)/(x2-x1)
# In GF: f(0) = y1*(-x2)/(x1-x2) + y2*(-x1)/(x2-x1)
#             = y1*x2/(x1+x2) + y2*x1/(x2+x1)  [since -a = a in GF(256)]

x1, y1 = 3, y3
x2, y2 = 15, y15

print(f"\nRecovering from shares: ({x1}, {y1}) and ({x2}, {y2})")

# Method 1: (y1*x2 + y2*x1) / (x1 + x2)
num = gf_add(gf_mul(y1, x2), gf_mul(y2, x1))
den = gf_add(x1, x2)
recovered1 = gf_div(num, den)
print(f"Method 1: {recovered1}")

# Method 2: Lagrange basis polynomials
# L1(0) = (0 - x2) / (x1 - x2) = x2 / (x1 + x2)
# L2(0) = (0 - x1) / (x2 - x1) = x1 / (x2 + x1)
L1_0 = gf_div(x2, gf_sub(x1, x2))
L2_0 = gf_div(x1, gf_sub(x2, x1))
recovered2 = gf_add(gf_mul(y1, L1_0), gf_mul(y2, L2_0))
print(f"Method 2: {recovered2}")

print(f"\nExpected: {secret}")
print(f"Match: {recovered1 == secret and recovered2 == secret}")

# Now test with the actual share data
print("\n" + "="*70)
print("Testing with actual share entropy (first byte only):")

x1, y1 = 3, 0xc4
x2, y2 = 15, 0x2b

print(f"Share 1: ({x1}, 0x{y1:02x})")
print(f"Share 2: ({x2}, 0x{y2:02x})")

num = gf_add(gf_mul(y1, x2), gf_mul(y2, x1))
den = gf_add(x1, x2)
recovered = gf_div(num, den)

print(f"Recovered byte: 0x{recovered:02x}")

# Let's also verify by computing what y values we'd get from different secrets
print("\nTrying different secrets to see which would produce these y values:")
for test_secret in range(256):
    # We need to find slope such that:
    # y1 = secret + slope * x1
    # y2 = secret + slope * x2
    # From first equation: slope = (y1 - secret) / x1
    # Check if it also satisfies second equation

    y1_minus_secret = gf_sub(y1, test_secret)
    slope = gf_div(y1_minus_secret, x1)

    # Verify with x2
    expected_y2 = gf_add(test_secret, gf_mul(slope, x2))

    if expected_y2 == y2:
        print(f"  Secret: 0x{test_secret:02x}, Slope: 0x{slope:02x}")
