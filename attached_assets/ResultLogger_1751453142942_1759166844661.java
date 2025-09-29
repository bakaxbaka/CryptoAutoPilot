import java.io.FileWriter;
import java.io.IOException;
import java.math.BigInteger;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Utility class for logging private key recovery results to a file
 */
public class ResultLogger {
    private static final String LOG_FILE = "recovery_results.log";
    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * Logs a recovery result to the log file
     * @param txid Transaction ID
     * @param privKey Recovered private key
     * @param wif WIF format of the private key
     * @param verified Whether the key was verified
     */
    public static void logResult(String txid, BigInteger privKey, String wif, boolean verified) {
        String timestamp = LocalDateTime.now().format(formatter);
        String logLine = String.format("[%s] TXID: %s\nPrivate Key: %s\nWIF: %s\nVerified: %s\n\n",
                timestamp, txid, privKey.toString(16), wif, verified ? "✅ Yes" : "❌ No");

        try (FileWriter fw = new FileWriter(LOG_FILE, true)) {
            fw.write(logLine);
            fw.flush();
        } catch (IOException e) {
            System.err.println("Failed to write log: " + e.getMessage());
        }
    }
    
    /**
     * Logs a detailed recovery result with additional information
     * @param txid1 First transaction ID
     * @param txid2 Second transaction ID  
     * @param privKey Recovered private key
     * @param wifCompressed Compressed WIF format
     * @param wifUncompressed Uncompressed WIF format
     * @param r Reused R value
     * @param verified Whether the key was verified
     */
    public static void logDetailedResult(String txid1, String txid2, BigInteger privKey, 
                                       String wifCompressed, String wifUncompressed, 
                                       BigInteger r, boolean verified) {
        String timestamp = LocalDateTime.now().format(formatter);
        String logLine = String.format(
            "[%s] Vulnerability Detected!\n" +
            "Transaction 1: %s\n" +
            "Transaction 2: %s\n" +
            "Reused R value: %s\n" +
            "Private Key (hex): %s\n" +
            "Private Key (decimal): %s\n" +
            "WIF Compressed: %s\n" +
            "WIF Uncompressed: %s\n" +
            "Verified: %s\n" +
            "----------------------------------------\n\n",
            timestamp, txid1, txid2, r.toString(16), 
            privKey.toString(16), privKey.toString(), 
            wifCompressed, wifUncompressed,
            verified ? "✅ Yes" : "❌ No"
        );

        try (FileWriter fw = new FileWriter(LOG_FILE, true)) {
            fw.write(logLine);
            fw.flush();
            System.out.println("Result logged to " + LOG_FILE);
        } catch (IOException e) {
            System.err.println("Failed to write log: " + e.getMessage());
        }
    }
}