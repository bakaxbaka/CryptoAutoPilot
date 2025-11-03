# Shamir Secret Sharing Vulnerability - Solution Summary

## Problem Analysis

We have 2 out of 3 required shares from a 3-of-5 Shamir Secret Sharing scheme that split a 12-word BIP-39 mnemonic. The target Bitcoin address has pubkey hash: `17f33b1f8ef28ac93e4b53753e3817d56a95750e`.

### Given Shares:
- **Share 1**: session cigar grape merry useful churn fatal thought very any arm unaware
- **Share 2**: clock fresh security field caution effort gorilla speed plastic common tomato echo

### Share Indices:
- Share 1: index = 3 (from last word & 0x0F)
- Share 2: index = 15 (from last word & 0x0F)

## Vulnerability

The implementation has a critical flaw: it uses a **linear polynomial (degree 1)** instead of the correct **quadratic polynomial (degree 2)** that a 3-of-5 threshold scheme requires.

### Expected vs Actual:
- **Expected** for 3-threshold: f(x) = a₀ + a₁x + a₂x² (degree 2, needs 3 points)
- **Actual vulnerability**: f(x) = a₀ + a₁x (degree 1, needs only 2 points)

This off-by-one error in the polynomial degree allows recovery with only 2 shares instead of the required 3.

## Technical Details

### GF(256) Arithmetic
The implementation uses standard AES/Rijndael GF(256) field with:
- Generator polynomial: x⁸ + x⁴ + x³ + x + 1 (0x11B)
- Multiplication by x (not x+1)

### Recovery Method
Using Lagrange interpolation at x=0:
```
f(0) = y₁ · L₁(0) + y₂ · L₂(0)
where L₁(0) = x₂/(x₁ + x₂) and L₂(0) = x₁/(x₁ + x₂) in GF(256)
```

Simplified:
```
f(0) = (y₁ · x₂ + y₂ · x₁) / (x₁ + x₂)
```

## Current Status

The recovery script (`recover_final.py`) implements the correct GF(256) arithmetic and Lagrange interpolation, but produces an incorrect mnemonic that doesn't match the target address.

### Recovered (incorrect):
- Mnemonic: `abandon abandon doctor abandon abandon letter advice abandon abandon absent`
- Entropy: `00000101000001010100000000000000`
- Pubkey hash: `4ed76f7af9475f76c9d1ce02bdb9c700b7cbfa39`

### Target:
- Pubkey hash: `17f33b1f8ef28ac93e4b53753e3817d56a95750e`

## Remaining Issues

The discrepancy suggests one of:
1. The polynomial is actually degree 2 (threshold properly implemented) and we need a different vulnerability exploit
2. The entropy extraction from shares is incorrect
3. The GF(256) field definition differs from what we're using
4. There's additional obfuscation or encoding we haven't accounted for

## Next Steps

1. Verify the exact GF(256) field used by the mnemonic tool
2. Check if there are additional transformations on the entropy
3. Investigate if the "bug report" vulnerability (coefficients never equal 255) can help narrow down candidates
4. Consider if we need to brute-force with additional constraints

## Files

- `recover_final.py` - Current recovery attempt with standard AES GF(256)
- `recover_v4.py` - Version using pybtc's non-standard GF(256) (x+1 multiplication)
- `analyze_shares.py` - Share structure analysis
- `test_gf256.py` - GF(256) arithmetic verification
