import hashlib, hmac, struct
from mnemonic import Mnemonic
import ecdsa
from itertools import product

# --- Constants ---
TARGET_PUBKEY_HASH_HEX = "17f33b1f8ef28ac93e4b53753e3817d56a95750e"
BIP39_SALT = b'mnemonic'
mnemo = Mnemonic("english")
wordlist = mnemo.wordlist

# --- GF(256) math ---
EXP, LOG = [0] * 512, [0] * 256
poly = 1
for i in range(255):
    EXP[i] = poly
    LOG[poly] = i
    poly <<= 1
    if poly & 0x100: poly ^= 0x11B
for i in range(255, 512): EXP[i] = EXP[i - 255]

def gf_add(a, b): return a ^ b
def gf_mul(a, b): return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]
def gf_div(a, b): return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]

# --- Lagrange interpolation in GF(256) ---
def interpolate(shares, x_target=0):
    """
    Interpolate polynomial at x_target using Lagrange interpolation
    shares: list of (x, y_bytes) tuples
    Returns: bytes of interpolated value
    """
    result = [0] * len(shares[0][1])
    
    for i in range(len(result)):
        for j, (xj, yj) in enumerate(shares):
            numerator = yj[i]
            denominator = 1
            
            for k, (xk, yk) in enumerate(shares):
                if j != k:
                    numerator = gf_mul(numerator, gf_add(x_target, xk))
                    denominator = gf_mul(denominator, gf_add(xj, xk))
            
            if denominator != 0:
                result[i] = gf_add(result[i], gf_div(numerator, denominator))
    
    return bytes(result)

# --- Extract share index ---
def extract_share_index(mnemonic: str):
    words = mnemonic.strip().split()
    last_word = words[-1]
    return wordlist.index(last_word) & 0x0F

# --- Convert mnemonic to entropy without checksum ---
def mnemonic_to_entropy_no_checksum(mnemonic: str) -> bytes:
    words = mnemonic.strip().split()
    bits = 0
    for word in words:
        bits = (bits << 11) | wordlist.index(word)
    checksum_len = (len(words) * 11) % 32
    entropy_bits = bits >> checksum_len
    return entropy_bits.to_bytes((len(words) * 11 - checksum_len) // 8, 'big')

# --- Shares ---
share1 = "session cigar grape merry useful churn fatal thought very any arm unaware"
share2 = "clock fresh security field caution effort gorilla speed plastic common tomato echo"

# --- Extract indices and entropy bytes ---
x1 = extract_share_index(share1)
x2 = extract_share_index(share2)
print(f"[*] Share indices: x1={x1}, x2={x2}")

y1 = mnemonic_to_entropy_no_checksum(share1)
y2 = mnemonic_to_entropy_no_checksum(share2)

# --- Derive address from mnemonic ---
def mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
    return hashlib.pbkdf2_hmac("sha512", mnemonic.encode(), BIP39_SALT + passphrase.encode(), 2048, dklen=64)

def hmac_sha512(key, data): 
    return hmac.new(key, data, hashlib.sha512).digest()

def CKD_priv(k_par, c_par, index):
    if index >= 0x80000000:
        data = b'\x00' + k_par + struct.pack('>L', index)
    else:
        pubkey = ecdsa.SigningKey.from_string(k_par, curve=ecdsa.SECP256k1).verifying_key
        data = pubkey.to_string("compressed") + struct.pack('>L', index)
    I = hmac_sha512(c_par, data)
    IL, IR = I[:32], I[32:]
    k_i = (int.from_bytes(IL, 'big') + int.from_bytes(k_par, 'big')) % ecdsa.SECP256k1.order
    return k_i.to_bytes(32, 'big'), IR

def derive_bip84_pubkey_hash(seed: bytes) -> str:
    k, c = hmac_sha512(b"Bitcoin seed", seed)[:32], hmac_sha512(b"Bitcoin seed", seed)[32:]
    for i in [84 + 0x80000000, 0 + 0x80000000, 0 + 0x80000000, 0, 0]:
        k, c = CKD_priv(k, c, i)
    pubkey = ecdsa.SigningKey.from_string(k, curve=ecdsa.SECP256k1).verifying_key
    x, y = pubkey.pubkey.point.x(), pubkey.pubkey.point.y()
    compressed_pubkey = bytes([0x02 + (y & 1)]) + x.to_bytes(32, 'big')
    return hashlib.new('ripemd160', hashlib.sha256(compressed_pubkey).digest()).hexdigest()

# Try different possible third share indices
print("[*] Trying different third share indices...")
for x3 in range(16):  # Try all possible 4-bit indices
    if x3 == x1 or x3 == x2:
        continue  # Skip if same as existing shares
    
    print(f"[*] Trying x3={x3}...")
    
    # Try a limited brute force on the third share's entropy
    # Since we have 16 bytes, we'll try some patterns first
    
    # Pattern 1: Try entropy similar to share1 or share2
    for base_entropy in [y1, y2]:
        for byte_idx in range(16):
            for byte_val in range(256):
                # Create candidate third share entropy
                y3_list = list(base_entropy)
                y3_list[byte_idx] = byte_val
                y3 = bytes(y3_list)
                
                # Interpolate secret
                shares = [(x1, y1), (x2, y2), (x3, y3)]
                try:
                    secret_bytes = interpolate(shares, 0)
                    valid_mnemonic = mnemo.to_mnemonic(secret_bytes)
                    
                    seed = mnemonic_to_seed(valid_mnemonic)
                    pubkey_hash = derive_bip84_pubkey_hash(seed)
                    
                    if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
                        print(f"\n[+] SUCCESS! Found valid mnemonic:")
                        print(f"[+] Third share index: x3={x3}")
                        print(f"[+] Mnemonic: {valid_mnemonic}")
                        exit(0)
                except Exception as e:
                    pass
    
    # Pattern 2: Try all zeros
    y3 = bytes([0] * 16)
    shares = [(x1, y1), (x2, y2), (x3, y3)]
    try:
        secret_bytes = interpolate(shares, 0)
        valid_mnemonic = mnemo.to_mnemonic(secret_bytes)
        seed = mnemonic_to_seed(valid_mnemonic)
        pubkey_hash = derive_bip84_pubkey_hash(seed)
        if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
            print(f"\n[+] SUCCESS! Found valid mnemonic:")
            print(f"[+] Third share index: x3={x3}")
            print(f"[+] Mnemonic: {valid_mnemonic}")
            exit(0)
    except:
        pass
    
    # Pattern 3: Try all 0xFF
    y3 = bytes([0xFF] * 16)
    shares = [(x1, y1), (x2, y2), (x3, y3)]
    try:
        secret_bytes = interpolate(shares, 0)
        valid_mnemonic = mnemo.to_mnemonic(secret_bytes)
        seed = mnemonic_to_seed(valid_mnemonic)
        pubkey_hash = derive_bip84_pubkey_hash(seed)
        if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
            print(f"\n[+] SUCCESS! Found valid mnemonic:")
            print(f"[+] Third share index: x3={x3}")
            print(f"[+] Mnemonic: {valid_mnemonic}")
            exit(0)
    except:
        pass

print("\n[-] Failed to find valid mnemonic with simple patterns")
print("[*] A more extensive brute force may be needed")
