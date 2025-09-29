import java.math.BigInteger;
import java.util.Arrays;

/**
 * Base58 encoding and decoding implementation
 * Used for Bitcoin address and private key (WIF) encoding
 */
public class Base58 {
    
    // Base58 alphabet (Bitcoin variant)
    private static final String ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    private static final char[] ALPHABET_CHARS = ALPHABET.toCharArray();
    private static final int[] ALPHABET_MAP = new int[128];
    
    static {
        // Initialize alphabet map for decoding
        Arrays.fill(ALPHABET_MAP, -1);
        for (int i = 0; i < ALPHABET_CHARS.length; i++) {
            ALPHABET_MAP[ALPHABET_CHARS[i]] = i;
        }
    }
    
    /**
     * Encodes a byte array to Base58 string
     * @param input The byte array to encode
     * @return Base58 encoded string
     */
    public String encode(byte[] input) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        if (input.length == 0) {
            return "";
        }
        
        // Count leading zeros
        int leadingZeros = 0;
        while (leadingZeros < input.length && input[leadingZeros] == 0) {
            leadingZeros++;
        }
        
        // Convert to BigInteger and encode
        BigInteger num = new BigInteger(1, input);
        StringBuilder encoded = new StringBuilder();
        
        // Convert to base 58
        while (num.compareTo(BigInteger.ZERO) > 0) {
            BigInteger[] divmod = num.divideAndRemainder(BigInteger.valueOf(58));
            encoded.insert(0, ALPHABET_CHARS[divmod[1].intValue()]);
            num = divmod[0];
        }
        
        // Add leading '1's for leading zeros
        for (int i = 0; i < leadingZeros; i++) {
            encoded.insert(0, '1');
        }
        
        return encoded.toString();
    }
    
    /**
     * Decodes a Base58 string to byte array
     * @param input The Base58 string to decode
     * @return Decoded byte array
     */
    public byte[] decode(String input) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        if (input.length() == 0) {
            return new byte[0];
        }
        
        // Count leading '1's
        int leadingOnes = 0;
        while (leadingOnes < input.length() && input.charAt(leadingOnes) == '1') {
            leadingOnes++;
        }
        
        // Convert string to BigInteger
        BigInteger num = BigInteger.ZERO;
        BigInteger base = BigInteger.valueOf(58);
        
        for (int i = leadingOnes; i < input.length(); i++) {
            char c = input.charAt(i);
            
            if (c >= ALPHABET_MAP.length || ALPHABET_MAP[c] == -1) {
                throw new IllegalArgumentException("Invalid Base58 character: " + c);
            }
            
            num = num.multiply(base).add(BigInteger.valueOf(ALPHABET_MAP[c]));
        }
        
        // Convert to byte array
        byte[] decoded = num.toByteArray();
        
        // Remove leading zero byte if present (from BigInteger.toByteArray())
        if (decoded.length > 1 && decoded[0] == 0) {
            byte[] temp = new byte[decoded.length - 1];
            System.arraycopy(decoded, 1, temp, 0, temp.length);
            decoded = temp;
        }
        
        // Add leading zeros for leading '1's
        if (leadingOnes > 0) {
            byte[] result = new byte[leadingOnes + decoded.length];
            Arrays.fill(result, 0, leadingOnes, (byte) 0);
            System.arraycopy(decoded, 0, result, leadingOnes, decoded.length);
            return result;
        }
        
        return decoded;
    }
    
    /**
     * Validates a Base58 string
     * @param input The string to validate
     * @return true if valid Base58, false otherwise
     */
    public boolean isValid(String input) {
        if (input == null || input.length() == 0) {
            return false;
        }
        
        try {
            for (char c : input.toCharArray()) {
                if (c >= ALPHABET_MAP.length || ALPHABET_MAP[c] == -1) {
                    return false;
                }
            }
            
            // Try to decode to verify
            decode(input);
            return true;
            
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Encodes a hex string to Base58
     * @param hex The hex string to encode
     * @return Base58 encoded string
     */
    public String encodeHex(String hex) {
        if (hex == null) {
            throw new IllegalArgumentException("Hex string cannot be null");
        }
        
        // Remove 0x prefix if present
        String cleanHex = hex.toLowerCase().replaceFirst("^0x", "");
        
        // Validate hex string
        if (!cleanHex.matches("[0-9a-f]*")) {
            throw new IllegalArgumentException("Invalid hex string: " + hex);
        }
        
        // Ensure even length
        if (cleanHex.length() % 2 != 0) {
            cleanHex = "0" + cleanHex;
        }
        
        // Convert hex to byte array
        byte[] bytes = new byte[cleanHex.length() / 2];
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] = (byte) Integer.parseInt(cleanHex.substring(i * 2, i * 2 + 2), 16);
        }
        
        return encode(bytes);
    }
    
    /**
     * Decodes Base58 to hex string
     * @param base58 The Base58 string to decode
     * @return Hex string representation
     */
    public String decodeToHex(String base58) {
        byte[] decoded = decode(base58);
        StringBuilder hex = new StringBuilder();
        
        for (byte b : decoded) {
            hex.append(String.format("%02x", b & 0xFF));
        }
        
        return hex.toString();
    }
    
    /**
     * Gets the Base58 alphabet
     * @return The alphabet string
     */
    public static String getAlphabet() {
        return ALPHABET;
    }
    
    /**
     * Checks if a character is valid in Base58
     * @param c The character to check
     * @return true if valid, false otherwise
     */
    public static boolean isValidChar(char c) {
        return c < ALPHABET_MAP.length && ALPHABET_MAP[c] != -1;
    }
}
