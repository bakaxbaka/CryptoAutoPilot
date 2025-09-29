import java.util.*;

/**
 * Represents a Bitcoin transaction with its signatures
 * Contains all the ECDSA signatures from the transaction inputs
 */
public class BitcoinTransaction {
    private String transactionId;
    private List<TransactionSignature> signatures;
    private Map<String, Object> metadata;
    
    /**
     * Constructor for BitcoinTransaction
     * @param transactionId The transaction ID (txid)
     */
    public BitcoinTransaction(String transactionId) {
        this.transactionId = transactionId;
        this.signatures = new ArrayList<>();
        this.metadata = new HashMap<>();
    }
    
    /**
     * Constructor with initial signatures
     * @param transactionId The transaction ID
     * @param signatures List of transaction signatures
     */
    public BitcoinTransaction(String transactionId, List<TransactionSignature> signatures) {
        this.transactionId = transactionId;
        this.signatures = new ArrayList<>(signatures);
        this.metadata = new HashMap<>();
    }
    
    /**
     * Adds a signature to the transaction
     * @param signature The signature to add
     */
    public void addSignature(TransactionSignature signature) {
        if (signature == null) {
            throw new IllegalArgumentException("Signature cannot be null");
        }
        this.signatures.add(signature);
    }
    
    /**
     * Adds a signature using hex string values
     * @param rHex R value as hex string
     * @param sHex S value as hex string
     * @param messageHashHex Message hash as hex string
     * @param inputIndex Input index for this signature
     */
    public void addSignature(String rHex, String sHex, String messageHashHex, int inputIndex) {
        TransactionSignature signature = new TransactionSignature(rHex, sHex, messageHashHex, inputIndex);
        addSignature(signature);
    }
    
    /**
     * Removes a signature at the specified index
     * @param index The index of the signature to remove
     * @return The removed signature, or null if index is invalid
     */
    public TransactionSignature removeSignature(int index) {
        if (index >= 0 && index < signatures.size()) {
            return signatures.remove(index);
        }
        return null;
    }
    
    /**
     * Gets a signature by its input index
     * @param inputIndex The input index to search for
     * @return The signature with the matching input index, or null if not found
     */
    public TransactionSignature getSignatureByInputIndex(int inputIndex) {
        return signatures.stream()
            .filter(sig -> sig.getInputIndex() == inputIndex)
            .findFirst()
            .orElse(null);
    }
    
    /**
     * Gets all signatures that have the same R value
     * @param rValue The R value to search for
     * @return List of signatures with matching R value
     */
    public List<TransactionSignature> getSignaturesByR(java.math.BigInteger rValue) {
        return signatures.stream()
            .filter(sig -> sig.getR().equals(rValue))
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Checks if the transaction has any reused R values
     * @return true if R values are reused, false otherwise
     */
    public boolean hasReusedRValues() {
        Set<java.math.BigInteger> rValues = new HashSet<>();
        for (TransactionSignature sig : signatures) {
            if (!rValues.add(sig.getR())) {
                return true; // R value already exists
            }
        }
        return false;
    }
    
    /**
     * Gets all unique R values in the transaction
     * @return Set of unique R values
     */
    public Set<java.math.BigInteger> getUniqueRValues() {
        Set<java.math.BigInteger> rValues = new HashSet<>();
        for (TransactionSignature sig : signatures) {
            rValues.add(sig.getR());
        }
        return rValues;
    }
    
    /**
     * Gets all reused R values (R values that appear more than once)
     * @return Set of reused R values
     */
    public Set<java.math.BigInteger> getReusedRValues() {
        Map<java.math.BigInteger, Integer> rCounts = new HashMap<>();
        
        // Count occurrences of each R value
        for (TransactionSignature sig : signatures) {
            rCounts.merge(sig.getR(), 1, Integer::sum);
        }
        
        // Return R values that appear more than once
        return rCounts.entrySet().stream()
            .filter(entry -> entry.getValue() > 1)
            .map(Map.Entry::getKey)
            .collect(HashSet::new, HashSet::add, HashSet::addAll);
    }
    
    /**
     * Validates the transaction structure
     * @return true if valid, false otherwise
     */
    public boolean isValid() {
        if (transactionId == null || transactionId.trim().isEmpty()) {
            return false;
        }
        
        if (signatures.isEmpty()) {
            return false;
        }
        
        // Check for duplicate input indices
        Set<Integer> inputIndices = new HashSet<>();
        for (TransactionSignature sig : signatures) {
            if (!inputIndices.add(sig.getInputIndex())) {
                return false; // Duplicate input index
            }
        }
        
        return true;
    }
    
    /**
     * Gets a summary of the transaction
     * @return Transaction summary string
     */
    public String getSummary() {
        int uniqueRCount = getUniqueRValues().size();
        int reusedRCount = getReusedRValues().size();
        
        return String.format(
            "Transaction %s: %d signatures, %d unique R values, %d reused R values",
            transactionId.substring(0, Math.min(8, transactionId.length())) + "...",
            signatures.size(),
            uniqueRCount,
            reusedRCount
        );
    }
    
    /**
     * Adds metadata to the transaction
     * @param key Metadata key
     * @param value Metadata value
     */
    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }
    
    /**
     * Gets metadata value by key
     * @param key Metadata key
     * @return Metadata value, or null if not found
     */
    public Object getMetadata(String key) {
        return metadata.get(key);
    }
    
    /**
     * Clears all signatures from the transaction
     */
    public void clearSignatures() {
        signatures.clear();
    }
    
    /**
     * Gets the number of signatures in the transaction
     * @return Number of signatures
     */
    public int getSignatureCount() {
        return signatures.size();
    }
    
    /**
     * Creates a copy of this transaction
     * @return A new BitcoinTransaction instance with copied data
     */
    public BitcoinTransaction copy() {
        BitcoinTransaction copy = new BitcoinTransaction(this.transactionId);
        copy.signatures.addAll(this.signatures);
        copy.metadata.putAll(this.metadata);
        return copy;
    }
    
    /**
     * String representation of the transaction
     * @return String representation
     */
    @Override
    public String toString() {
        return String.format("BitcoinTransaction{id=%s, signatures=%d, vulnerable=%s}",
            transactionId,
            signatures.size(),
            hasReusedRValues()
        );
    }
    
    // Getters and setters
    public String getTransactionId() {
        return transactionId;
    }
    
    public void setTransactionId(String transactionId) {
        this.transactionId = transactionId;
    }
    
    public List<TransactionSignature> getSignatures() {
        return new ArrayList<>(signatures); // Return copy to prevent external modification
    }
    
    public void setSignatures(List<TransactionSignature> signatures) {
        this.signatures = new ArrayList<>(signatures);
    }
    
    public Map<String, Object> getMetadata() {
        return new HashMap<>(metadata); // Return copy to prevent external modification
    }
}
