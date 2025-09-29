import java.math.BigInteger;

/**
 * Represents an ECDSA signature from a Bitcoin transaction input
 * Contains the R, S values and the message hash that was signed
 */
public class TransactionSignature {
    private BigInteger r;
    private BigInteger s;
    private BigInteger messageHash;
    private int inputIndex;
    
    /**
     * Constructor for TransactionSignature
     * @param r The R component of the ECDSA signature
     * @param s The S component of the ECDSA signature
     * @param messageHash The hash of the message that was signed
     * @param inputIndex The index of this input in the transaction
     */
    public TransactionSignature(BigInteger r, BigInteger s, BigInteger messageHash, int inputIndex) {
        this.r = r;
        this.s = s;
        this.messageHash = messageHash;
        this.inputIndex = inputIndex;
        
        validateSignature();
    }
    
    /**
     * Constructor that accepts hex strings
     * @param rHex R value as hexadecimal string
     * @param sHex S value as hexadecimal string
     * @param messageHashHex Message hash as hexadecimal string
     * @param inputIndex The index of this input in the transaction
     */
    public TransactionSignature(String rHex, String sHex, String messageHashHex, int inputIndex) {
        this.r = parseHexValue(rHex);
        this.s = parseHexValue(sHex);
        this.messageHash = parseHexValue(messageHashHex);
        this.inputIndex = inputIndex;
        
        validateSignature();
    }
    
    /**
     * Parses a hexadecimal string to BigInteger, handling various formats
     * @param hexValue The hex string to parse
     * @return BigInteger representation
     */
    private BigInteger parseHexValue(String hexValue) {
        if (hexValue == null || hexValue.trim().isEmpty()) {
            throw new IllegalArgumentException("Hex value cannot be null or empty");
        }
        
        String normalized = hexValue.trim();
        
        // Remove 0x prefix if present
        if (normalized.toLowerCase().startsWith("0x")) {
            normalized = normalized.substring(2);
        }
        
        // Validate hex characters
        if (!normalized.matches("[0-9a-fA-F]+")) {
            throw new IllegalArgumentException("Invalid hex string: " + hexValue);
        }
        
        try {
            return new BigInteger(normalized, 16);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Failed to parse hex value: " + hexValue, e);
        }
    }
    
    /**
     * Validates the signature components
     * @throws IllegalArgumentException if any component is invalid
     */
    private void validateSignature() {
        if (r == null || s == null || messageHash == null) {
            throw new IllegalArgumentException("Signature components cannot be null");
        }
        
        if (r.equals(BigInteger.ZERO)) {
            throw new IllegalArgumentException("R value cannot be zero");
        }
        
        if (s.equals(BigInteger.ZERO)) {
            throw new IllegalArgumentException("S value cannot be zero");
        }
        
        if (messageHash.compareTo(BigInteger.ZERO) < 0) {
            throw new IllegalArgumentException("Message hash cannot be negative");
        }
        
        // Check if values are within valid range for secp256k1
        BigInteger curveOrder = ECDSAAttack.CURVE_ORDER;
        
        // R and S values should be positive and less than the curve order for valid ECDSA signatures
        // However, for educational purposes with real-world examples, we allow some flexibility
        if (r.compareTo(BigInteger.ZERO) <= 0) {
            throw new IllegalArgumentException("R value must be positive");
        }
        
        if (s.compareTo(BigInteger.ZERO) <= 0) {
            throw new IllegalArgumentException("S value must be positive");
        }
        
        // Allow values up to curve order for legitimate signatures
        // Values larger than curve order will be handled by modular arithmetic
        if (r.compareTo(curveOrder) >= 0) {
            System.out.println("Warning: R value exceeds curve order, will use modular arithmetic");
        }
        
        if (s.compareTo(curveOrder) >= 0) {
            System.out.println("Warning: S value exceeds curve order, will use modular arithmetic");
        }
    }
    
    /**
     * Creates a copy of this signature with a different input index
     * @param newIndex The new input index
     * @return A new TransactionSignature instance
     */
    public TransactionSignature withInputIndex(int newIndex) {
        return new TransactionSignature(this.r, this.s, this.messageHash, newIndex);
    }
    
    /**
     * Checks if this signature has the same R value as another signature
     * @param other The other signature to compare with
     * @return true if R values are equal, false otherwise
     */
    public boolean hasSameR(TransactionSignature other) {
        return this.r.equals(other.r);
    }
    
    /**
     * Returns a string representation of the signature for debugging
     * @return String representation
     */
    @Override
    public String toString() {
        return String.format("TransactionSignature{input=%d, r=%s, s=%s, m=%s}", 
            inputIndex,
            r.toString(16).substring(0, Math.min(16, r.toString(16).length())) + "...",
            s.toString(16).substring(0, Math.min(16, s.toString(16).length())) + "...",
            messageHash.toString(16).substring(0, Math.min(16, messageHash.toString(16).length())) + "..."
        );
    }
    
    /**
     * Returns a detailed string representation for logging
     * @return Detailed string representation
     */
    public String toDetailedString() {
        return String.format("TransactionSignature{\n" +
            "  inputIndex=%d,\n" +
            "  r=%s,\n" +
            "  s=%s,\n" +
            "  messageHash=%s\n" +
            "}", inputIndex, r.toString(16), s.toString(16), messageHash.toString(16));
    }
    
    /**
     * Equals method for comparing signatures
     * @param obj Object to compare with
     * @return true if signatures are equal, false otherwise
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        TransactionSignature that = (TransactionSignature) obj;
        return inputIndex == that.inputIndex &&
               r.equals(that.r) &&
               s.equals(that.s) &&
               messageHash.equals(that.messageHash);
    }
    
    /**
     * Hash code for the signature
     * @return Hash code
     */
    @Override
    public int hashCode() {
        return r.hashCode() ^ s.hashCode() ^ messageHash.hashCode() ^ Integer.hashCode(inputIndex);
    }
    
    // Getters
    public BigInteger getR() {
        return r;
    }
    
    public BigInteger getS() {
        return s;
    }
    
    public BigInteger getMessageHash() {
        return messageHash;
    }
    
    public int getInputIndex() {
        return inputIndex;
    }
    
    // Setters (if needed for mutable operations)
    public void setInputIndex(int inputIndex) {
        this.inputIndex = inputIndex;
    }
}
