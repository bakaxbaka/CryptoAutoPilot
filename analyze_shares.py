#!/usr/bin/env python3
"""
Analyze the share structure to understand the encoding
"""

from mnemonic import Mnemonic

mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

SHARE_1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
SHARE_2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"

def analyze_share(share_str, name):
    print(f"\n{'='*70}")
    print(f"Analyzing {name}")
    print(f"{'='*70}")
    
    words = share_str.strip().split()
    print(f"Words: {words}")
    print(f"Number of words: {len(words)}")
    
    # Get word indices
    indices = [wordlist.index(w) for w in words]
    print(f"\nWord indices:")
    for i, (word, idx) in enumerate(zip(words, indices)):
        print(f"  {i+1:2d}. {word:12s} -> {idx:4d} (0x{idx:04x}) (binary: {idx:011b})")
    
    # Analyze last word
    last_word = words[-1]
    last_idx = wordlist.index(last_word)
    print(f"\nLast word analysis:")
    print(f"  Word: {last_word}")
    print(f"  Index: {last_idx} (0x{last_idx:04x})")
    print(f"  Binary: {last_idx:011b}")
    print(f"  Lower 4 bits: {last_idx & 0x0F} (0x{last_idx & 0x0F:x})")
    print(f"  Lower 5 bits: {last_idx & 0x1F} (0x{last_idx & 0x1F:x})")
    print(f"  Bits 4-7: {(last_idx >> 4) & 0x0F} (0x{(last_idx >> 4) & 0x0F:x})")
    
    # Convert to full bit string
    bits = 0
    for word in words:
        word_index = wordlist.index(word)
        bits = (bits << 11) | word_index
    
    total_bits = len(words) * 11
    print(f"\nTotal bits: {total_bits}")
    print(f"Bit string: {bits:0{total_bits}b}")
    
    # Extract entropy (without checksum)
    checksum_len = total_bits % 32
    entropy_bits = bits >> checksum_len
    entropy_byte_len = (total_bits - checksum_len) // 8
    entropy = entropy_bits.to_bytes(entropy_byte_len, 'big')
    
    print(f"\nChecksum length: {checksum_len} bits")
    print(f"Entropy length: {total_bits - checksum_len} bits ({entropy_byte_len} bytes)")
    print(f"Entropy (hex): {entropy.hex()}")
    print(f"Entropy (bytes): {list(entropy)}")

analyze_share(SHARE_1, "Share 1")
analyze_share(SHARE_2, "Share 2")

# Try to understand the relationship
print(f"\n{'='*70}")
print("Relationship Analysis")
print(f"{'='*70}")

words1 = SHARE_1.strip().split()
words2 = SHARE_2.strip().split()

idx1 = wordlist.index(words1[-1])
idx2 = wordlist.index(words2[-1])

print(f"\nLast word indices:")
print(f"  Share 1: {idx1} (0x{idx1:04x})")
print(f"  Share 2: {idx2} (0x{idx2:04x})")
print(f"  XOR: {idx1 ^ idx2} (0x{(idx1 ^ idx2):04x})")
print(f"  Difference: {abs(idx1 - idx2)}")

# Check if the index might be encoded differently
print(f"\nPossible index encodings:")
print(f"  Share 1 - Lower 4 bits: {idx1 & 0x0F}")
print(f"  Share 2 - Lower 4 bits: {idx2 & 0x0F}")
print(f"  Share 1 - Lower 5 bits: {idx1 & 0x1F}")
print(f"  Share 2 - Lower 5 bits: {idx2 & 0x1F}")
print(f"  Share 1 - Upper 4 bits: {(idx1 >> 7) & 0x0F}")
print(f"  Share 2 - Upper 4 bits: {(idx2 >> 7) & 0x0F}")
