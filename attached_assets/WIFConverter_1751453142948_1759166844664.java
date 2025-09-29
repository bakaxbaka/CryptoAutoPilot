import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Converts private keys to Wallet Import Format (WIF)
 * Supports both compressed and uncompressed formats
 */
public class WIFConverter {
    
    // Bitcoin mainnet version byte for WIF
    private static final byte MAINNET_VERSION = (byte) 0x80;
    
    // Compression flag
    private static final byte COMPRESSION_FLAG = (byte) 0x01;
    
    private Base58 base58;
    
    public WIFConverter() {
        this.base58 = new Base58();
    }
    
    /**
     * Converts a private key to WIF format
     * @param privateKey The private key as BigInteger
     * @param compressed Whether to generate compressed WIF
     * @return The WIF string
     */
    public String toWIF(BigInteger privateKey, boolean compressed) {
        if (privateKey == null) {
            throw new IllegalArgumentException("Private key cannot be null");
        }
        
        if (privateKey.equals(BigInteger.ZERO)) {
            throw new IllegalArgumentException("Private key cannot be zero");
        }
        
        if (privateKey.compareTo(ECDSAAttack.CURVE_ORDER) >= 0) {
            throw new IllegalArgumentException("Private key exceeds curve order");
        }
        
        try {
            // Convert private key to 32-byte array
            byte[] privateKeyBytes = toByteArray32(privateKey);
            
            // Create payload: version + private key + compression flag (if compressed)
            byte[] payload;
            if (compressed) {
                payload = new byte[34]; // 1 + 32 + 1
                payload[0] = MAINNET_VERSION;
                System.arraycopy(privateKeyBytes, 0, payload, 1, 32);
                payload[33] = COMPRESSION_FLAG;
            } else {
                payload = new byte[33]; // 1 + 32
                payload[0] = MAINNET_VERSION;
                System.arraycopy(privateKeyBytes, 0, payload, 1, 32);
            }
            
            // Calculate checksum (first 4 bytes of double SHA-256)
            byte[] checksum = calculateChecksum(payload);
            
            // Create final array: payload + checksum
            byte[] finalArray = new byte[payload.length + 4];
            System.arraycopy(payload, 0, finalArray, 0, payload.length);
            System.arraycopy(checksum, 0, finalArray, payload.length, 4);
            
            // Base58 encode
            return base58.encode(finalArray);
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to convert private key to WIF: " + e.getMessage(), e);
        }
    }
    
    /**
     * Converts a private key from WIF format back to BigInteger
     * @param wif The WIF string
     * @return WIFDecodeResult containing the private key and compression flag
     */
    public WIFDecodeResult fromWIF(String wif) {
        if (wif == null || wif.trim().isEmpty()) {
            throw new IllegalArgumentException("WIF string cannot be null or empty");
        }
        
        try {
            // Base58 decode
            byte[] decoded = base58.decode(wif.trim());
            
            // Validate length
            if (decoded.length != 37 && decoded.length != 38) {
                throw new IllegalArgumentException("Invalid WIF length");
            }
            
            // Check version byte
            if (decoded[0] != MAINNET_VERSION) {
                throw new IllegalArgumentException("Invalid WIF version byte");
            }
            
            // Determine if compressed
            boolean compressed = decoded.length == 38;
            
            // Extract payload and checksum
            int payloadLength = decoded.length - 4;
            byte[] payload = new byte[payloadLength];
            byte[] checksum = new byte[4];
            
            System.arraycopy(decoded, 0, payload, 0, payloadLength);
            System.arraycopy(decoded, payloadLength, checksum, 0, 4);
            
            // Verify checksum
            byte[] calculatedChecksum = calculateChecksum(payload);
            if (!java.util.Arrays.equals(checksum, calculatedChecksum)) {
                throw new IllegalArgumentException("Invalid WIF checksum");
            }
            
            // Extract private key (skip version byte, stop before compression flag if present)
            byte[] privateKeyBytes = new byte[32];
            System.arraycopy(payload, 1, privateKeyBytes, 0, 32);
            
            // Convert to BigInteger
            BigInteger privateKey = new BigInteger(1, privateKeyBytes);
            
            // Validate private key
            if (privateKey.equals(BigInteger.ZERO)) {
                throw new IllegalArgumentException("Invalid private key: zero");
            }
            
            if (privateKey.compareTo(ECDSAAttack.CURVE_ORDER) >= 0) {
                throw new IllegalArgumentException("Invalid private key: exceeds curve order");
            }
            
            return new WIFDecodeResult(privateKey, compressed);
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to decode WIF: " + e.getMessage(), e);
        }
    }
    
    /**
     * Converts BigInteger to 32-byte array (big-endian, zero-padded)
     * @param value The BigInteger to convert
     * @return 32-byte array
     */
    private byte[] toByteArray32(BigInteger value) {
        byte[] bytes = value.toByteArray();
        
        if (bytes.length == 32) {
            return bytes;
        } else if (bytes.length == 33 && bytes[0] == 0) {
            // Remove leading zero byte
            byte[] result = new byte[32];
            System.arraycopy(bytes, 1, result, 0, 32);
            return result;
        } else if (bytes.length < 32) {
            // Pad with leading zeros
            byte[] result = new byte[32];
            System.arraycopy(bytes, 0, result, 32 - bytes.length, bytes.length);
            return result;
        } else {
            throw new IllegalArgumentException("Private key too large for 32 bytes");
        }
    }
    
    /**
     * Calculates the checksum for WIF (first 4 bytes of double SHA-256)
     * @param data The data to hash
     * @return 4-byte checksum
     */
    private byte[] calculateChecksum(byte[] data) {
        try {
            MessageDigest sha256 = MessageDigest.getInstance("SHA-256");
            
            // First SHA-256
            byte[] hash1 = sha256.digest(data);
            
            // Second SHA-256
            sha256.reset();
            byte[] hash2 = sha256.digest(hash1);
            
            // Return first 4 bytes
            byte[] checksum = new byte[4];
            System.arraycopy(hash2, 0, checksum, 0, 4);
            return checksum;
            
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 algorithm not available", e);
        }
    }
    
    /**
     * Validates a WIF string format
     * @param wif The WIF string to validate
     * @return true if valid format, false otherwise
     */
    public boolean isValidWIF(String wif) {
        try {
            fromWIF(wif);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Determines if a WIF string represents a compressed key
     * @param wif The WIF string
     * @return true if compressed, false if uncompressed
     */
    public boolean isCompressedWIF(String wif) {
        try {
            WIFDecodeResult result = fromWIF(wif);
            return result.isCompressed();
        } catch (Exception e) {
            throw new RuntimeException("Invalid WIF string", e);
        }
    }
    
    /**
     * Result class for WIF decoding
     */
    public static class WIFDecodeResult {
        private final BigInteger privateKey;
        private final boolean compressed;
        
        public WIFDecodeResult(BigInteger privateKey, boolean compressed) {
            this.privateKey = privateKey;
            this.compressed = compressed;
        }
        
        public BigInteger getPrivateKey() {
            return privateKey;
        }
        
        public boolean isCompressed() {
            return compressed;
        }
        
        @Override
        public String toString() {
            return String.format("WIFDecodeResult{privateKey=%s, compressed=%s}", 
                privateKey.toString(16), compressed);
        }
    }
}
