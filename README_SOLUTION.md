# Shamir Secret Sharing Mnemonic Recovery

## Problem Analysis

You have two shares from a 3-of-5 Shamir Secret Sharing scheme:
- **Share 1**: session cigar grape merry useful churn fatal thought very any arm unaware
- **Share 2**: clock fresh security field caution effort gorilla speed plastic common tomato echo

Target Bitcoin address pubkey hash: `17f33b1f8ef28ac93e4b53753e3817d56a95750e`

## Vulnerability

The bitaps implementation has a known vulnerability where it uses a linear polynomial (degree 1) instead of a degree-2 polynomial for a 3-of-5 threshold scheme. This means only 2 shares should be sufficient to recover the secret.

## Investigation Results

I've created several recovery scripts that:

1. **Analyzed the share structure** (`analyze_shares.py`)
   - Share 1 index (from last word): 3
   - Share 2 index (from last word): 15
   - Entropy extracted successfully

2. **Implemented proper GF(256) arithmetic** (`final_recovery.py`)
   - Correct Galois Field operations
   - Lagrange interpolation at x=0

3. **Performed exhaustive search** 
   - Tested all 65,280 possible index combinations (1-255 for each share)
   - No combination produced the target address

## Possible Issues

The exhaustive search suggests one of the following:

1. **The target address might be derived differently**
   - Perhaps not using BIP-84 (m/84'/0'/0'/0/0)
   - Could be BIP-44 (m/44'/0'/0'/0/0) or another derivation path
   - Could have a passphrase

2. **The shares might be encoded differently**
   - The index extraction method might be different
   - The entropy extraction might need adjustment

3. **Additional information is needed**
   - A third share
   - The exact implementation details
   - Passphrase or salt

## Scripts Created

- `recover_mnemonic.py` - Initial recovery attempt
- `advanced_recovery.py` - Multiple recovery methods
- `analyze_shares.py` - Share structure analysis
- `full_recovery.py` - Complete Lagrange interpolation
- `fast_search.py` - Optimized brute force
- `final_recovery.py` - Exhaustive search with progress

## Next Steps

To proceed, we need:

1. **Verify the derivation path** - Is it definitely m/84'/0'/0'/0/0?
2. **Check for passphrase** - Is there a BIP-39 passphrase?
3. **Confirm the implementation** - Can you provide the exact code used to generate the shares?
4. **Third share** - If available, provide the third share for verification

## Running the Scripts

```bash
# Analyze share structure
python3 analyze_shares.py

# Try recovery with exhaustive search
python3 final_recovery.py

# Quick analysis
python3 recover_mnemonic.py
```

All scripts are ready to use and have been tested.
