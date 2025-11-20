import hashlib, hmac, struct
from mnemonic import Mnemonic
import ecdsa

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
def gf_sub(a, b): return a ^ b
def gf_mul(a, b): return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]
def gf_div(a, b): return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]

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

y1 = list(mnemonic_to_entropy_no_checksum(share1))
y2 = list(mnemonic_to_entropy_no_checksum(share2))

# --- Interpolate secret at x=0 ---
def compute_secret_byte(y1, y2, x1, x2):
    # s = (y1 * x2 + y2 * x1) / (x1 + x2)
    numerator = gf_add( gf_mul(y1, x2), gf_mul(y2, x1) )
    denominator = gf_add(x1, x2)
    return gf_div(numerator, denominator)

secret_bytes = bytes([compute_secret_byte(y1[i], y2[i], x1, x2) for i in range(len(y1))])
valid_mnemonic = mnemo.to_mnemonic(secret_bytes)
print(f"[*] Reconstructed mnemonic: {valid_mnemonic}")

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

seed = mnemonic_to_seed(valid_mnemonic)
pubkey_hash = derive_bip84_pubkey_hash(seed)
print(f"[*] Derived pubkey hash: {pubkey_hash}")
print(f"[*] Target pubkey hash:  {TARGET_PUBKEY_HASH_HEX}")

if pubkey_hash == TARGET_PUBKEY_HASH_HEX:
    print(f"\n[+] SUCCESS! Valid mnemonic found:")
    print(f"[+] {valid_mnemonic}")
else:
    print("\n[-] Failed to find correct mnemonic.")
    print(f"[-] Hash mismatch: {pubkey_hash} != {TARGET_PUBKEY_HASH_HEX}")
