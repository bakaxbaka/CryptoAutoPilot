#!/usr/bin/env python3
"""
Analyze the share structure to understand how indices are encoded
"""

from mnemonic import Mnemonic

mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

share1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
share2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"

print("=" * 70)
print("Share Analysis")
print("=" * 70)

def analyze_share(share_text, share_num):
    print(f"\n{'='*70}")
    print(f"Share {share_num}: {share_text}")
    print(f"{'='*70}")

    words = share_text.strip().split()
    print(f"\nNumber of words: {len(words)}")

    # Convert to bit string
    bits = 0
    for i, word in enumerate(words):
        word_index = wordlist.index(word)
        bits = (bits << 11) | word_index
        print(f"Word {i+1:2d}: '{word:12s}' -> index {word_index:4d} (0x{word_index:03x}) = {bin(word_index)[2:].zfill(11)}")

    total_bits = len(words) * 11
    print(f"\nTotal bits: {total_bits}")

    # For 12 words: 132 bits total = 128 entropy + 4 checksum
    # The share index is in the checksum bits (last 4 bits)
    entropy_bits = 128
    checksum_bits = total_bits - entropy_bits

    print(f"Expected entropy bits: {entropy_bits}")
    print(f"Expected checksum bits: {checksum_bits}")

    # Extract entropy and checksum
    entropy_int = bits >> checksum_bits
    checksum_int = bits & ((1 << checksum_bits) - 1)

    print(f"\nEntropy (first {entropy_bits} bits): {hex(entropy_int)}")
    print(f"Checksum/Index (last {checksum_bits} bits): {checksum_int} (0x{checksum_int:x})")

    entropy_bytes = entropy_int.to_bytes(16, 'big')
    print(f"Entropy bytes: {entropy_bytes.hex()}")

    # Last word analysis
    last_word = words[-1]
    last_word_index = wordlist.index(last_word)
    print(f"\nLast word: '{last_word}' -> index {last_word_index} (0x{last_word_index:03x}) = {bin(last_word_index)[2:].zfill(11)}")
    print(f"Last word lower 4 bits: {last_word_index & 0x0F}")
    print(f"Last word lower 5 bits: {last_word_index & 0x1F}")
    print(f"Last word lower 8 bits: {last_word_index & 0xFF}")

    return {
        'words': words,
        'bits': bits,
        'entropy_bytes': entropy_bytes,
        'checksum_int': checksum_int,
        'last_word_index': last_word_index
    }

info1 = analyze_share(share1, 1)
info2 = analyze_share(share2, 2)

print("\n" + "=" * 70)
print("Potential share indices:")
print("=" * 70)
print(f"Share 1 - last word & 0x0F: {info1['last_word_index'] & 0x0F}")
print(f"Share 2 - last word & 0x0F: {info2['last_word_index'] & 0x0F}")
print(f"\nShare 1 - checksum value: {info1['checksum_int']}")
print(f"Share 2 - checksum value: {info2['checksum_int']}")
