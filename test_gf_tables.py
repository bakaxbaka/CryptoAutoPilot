#!/usr/bin/env python3
"""Compare GF(256) table generation methods"""

# User's method (from the code they provided)
def method1():
    EXP, LOG = [0] * 512, [0] * 256
    poly = 1
    for i in range(255):
        EXP[i] = poly
        LOG[poly] = i
        poly <<= 1  # Just left shift
        if poly & 0x100: poly ^= 0x11B
    for i in range(255, 512): EXP[i] = EXP[i - 255]
    return EXP, LOG

# PyBTC method
def method2():
    EXP, LOG = [0] * 512, [0] * 256
    poly = 1
    for i in range(255):
        EXP[i] = poly
        LOG[poly] = i
        poly = (poly << 1) ^ poly  # Left shift XOR poly
        if poly & 0x100: poly ^= 0x11B
    for i in range(255, 512): EXP[i] = EXP[i - 255]
    return EXP, LOG

EXP1, LOG1 = method1()
EXP2, LOG2 = method2()

print("Comparing first 20 EXP values:")
for i in range(20):
    print(f"  EXP[{i:2d}]: method1={EXP1[i]:3d}, method2={EXP2[i]:3d}, {'✓' if EXP1[i]==EXP2[i] else '✗ DIFF'}")

print("\nComparing LOG values for first 20 non-zero entries:")
count = 0
for i in range(256):
    if LOG1[i] > 0 or LOG2[i] > 0:
        if count < 20:
            print(f"  LOG[{i:3d}]: method1={LOG1[i]:3d}, method2={LOG2[i]:3d}, {'✓' if LOG1[i]==LOG2[i] else '✗ DIFF'}")
            count += 1
