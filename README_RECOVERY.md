# Shamir Secret Sharing Mnemonic Recovery

## Overview

This repository contains tools to recover a BIP-39 mnemonic from Shamir Secret Sharing shares, exploiting the vulnerability in the bitaps implementation where a 3-of-5 threshold scheme uses a linear polynomial (degree 1) instead of degree 2, allowing recovery with only 2 shares.

## The Problem

You have two shares from a 3-of-5 Shamir Secret Sharing scheme:

**Share 1**: `session cigar grape merry useful churn fatal thought very any arm unaware`  
**Share 2**: `clock fresh security field caution effort gorilla speed plastic common tomato echo`

**Target Address**: `bc1q17f33b1f8ef28ac93e4b53753e3817d56a95750e`  
**Pubkey Hash**: `17f33b1f8ef28ac93e4b53753e3817d56a95750e`

## Quick Start

### Basic Recovery (Fast Mode)
```bash
python3 recover.py --fast
```

This tests the most common index combinations (1-16) with standard BIP paths.

### Full Recovery
```bash
python3 recover.py
```

This performs an exhaustive search of all 65,280 possible index combinations.

### With Passphrase
```bash
python3 recover.py --passphrase "your passphrase here"
```

If the mnemonic is protected with a BIP-39 passphrase, provide it here.

## Available Scripts

### Main Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `recover.py` | **Main recovery tool** - User-friendly with options | `python3 recover.py [--fast] [--passphrase PASS]` |
| `final_recovery.py` | Exhaustive search with progress tracking | `python3 final_recovery.py` |
| `analyze_shares.py` | Analyze share structure and encoding | `python3 analyze_shares.py` |

### Analysis Scripts

| Script | Description |
|--------|-------------|
| `test_derivation_paths.py` | Test different BIP derivation paths |
| `advanced_recovery.py` | Multiple recovery methods |
| `full_recovery.py` | Complete Lagrange interpolation |

## How It Works

### 1. Share Structure

Each share is a 12-word BIP-39 mnemonic where:
- First 11 words + first 7 bits of 12th word = entropy (128 bits)
- Last 4 bits of 12th word = checksum
- Share index is encoded in the last word

### 2. GF(256) Arithmetic

Shamir Secret Sharing uses Galois Field GF(2^8) arithmetic:
- Polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
- Operations: addition (XOR), multiplication, division

### 3. Lagrange Interpolation

To recover the secret at x=0 from two shares (x₁, y₁) and (x₂, y₂):

```
f(0) = (y₁ × x₂ ⊕ y₂ × x₁) / (x₁ ⊕ x₂)
```

Where ⊕ is XOR and ×, / are GF(256) operations.

### 4. Address Derivation

The recovered entropy is converted to a valid BIP-39 mnemonic, then:
1. Mnemonic → Seed (PBKDF2-HMAC-SHA512)
2. Seed → Master Key (HMAC-SHA512 with "Bitcoin seed")
3. Master Key → Derived Key (BIP-32 path)
4. Derived Key → Public Key (ECDSA secp256k1)
5. Public Key → Address Hash (SHA256 + RIPEMD160)

## Investigation Results

### What Was Tested

✅ All 65,280 possible share index combinations (1-255 for each share)  
✅ Multiple BIP derivation paths (BIP-44, BIP-49, BIP-84)  
✅ Proper GF(256) Galois Field arithmetic  
✅ Correct Lagrange interpolation  
✅ Valid BIP-39 mnemonic generation with checksum  

### Results

❌ **No combination produced the target address**

This suggests one of the following:

1. **Passphrase Protection** - The mnemonic uses a BIP-39 passphrase
2. **Custom Derivation Path** - Non-standard BIP path
3. **Three Shares Required** - Despite the vulnerability, 3 shares may be needed
4. **Different Encoding** - Share indices encoded differently
5. **Incorrect Target** - The provided address doesn't match these shares

## Technical Details

### Share Analysis

**Share 1**:
- Last word: "unaware" (index 1891)
- Share index: 3 (lower 4 bits)
- Entropy: `c4451d9745defe5194e707f2e1442ef6`

**Share 2**:
- Last word: "echo" (index 559)
- Share index: 15 (lower 4 bits)
- Entropy: `2b6b9b0b2af24a8d592e88a605cf9022`

### Tested Derivation Paths

- **BIP-44** (Legacy): m/44'/0'/0'/0/0
- **BIP-49** (SegWit): m/49'/0'/0'/0/0
- **BIP-84** (Native SegWit): m/84'/0'/0'/0/0

### Performance

- **Speed**: ~339 checks per second
- **Full search time**: ~3 minutes
- **Total combinations**: 65,280

## Vulnerability References

This recovery exploits known vulnerabilities in Shamir Secret Sharing implementations:

1. **Armory Wallet**: https://btcarmory.com/fragmented-backup-vuln/
2. **HTC Exodus**: https://donjon.ledger.com/Stealing-all-HTC-Exodus-users/
3. **Bitaps Issue**: https://github.com/bitaps-com/pybtc/issues/23

The vulnerability: Using a degree-1 polynomial for a 3-of-5 threshold allows recovery with only 2 shares.

## Next Steps

To successfully recover the mnemonic, please provide:

### Option 1: Passphrase
If you have a BIP-39 passphrase:
```bash
python3 recover.py --passphrase "your passphrase"
```

### Option 2: Custom Path
If using a non-standard derivation path, modify `recover.py`:
```python
paths = [
    ('Custom path', [0x80000000, 0x80000000, 0, 0]),  # Your path here
]
```

### Option 3: Third Share
If you have a third share, add it to the script and use 3-share Lagrange interpolation.

### Option 4: Verify Target
Double-check that the target address `bc1q17f33b1f8ef28ac93e4b53753e3817d56a95750e` is correct.

## Dependencies

```bash
pip install mnemonic ecdsa
```

Or use the existing `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Files Created

- `recover.py` - Main recovery tool (recommended)
- `final_recovery.py` - Exhaustive search
- `analyze_shares.py` - Share analysis
- `test_derivation_paths.py` - Path testing
- `FINDINGS.md` - Detailed investigation report
- `README_RECOVERY.md` - This file

## Support

If you need help or have additional information:

1. Check `FINDINGS.md` for detailed analysis
2. Run `python3 analyze_shares.py` to see share structure
3. Try `python3 recover.py --fast` first
4. Provide any missing information (passphrase, path, third share)

## Disclaimer

This tool is for educational and recovery purposes only. Use responsibly and only on your own funds.

---

**Status**: Ready to use. Awaiting additional information (passphrase, custom path, or third share) to complete recovery.
