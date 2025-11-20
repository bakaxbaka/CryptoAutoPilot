import unittest

from main import BitcoinBlockAnalyzer


class SighashComputationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = BitcoinBlockAnalyzer()

    def _prepare_prevout(self, prev_txid: str, vout: int, script_pubkey_hex: str, value: int = 0) -> None:
        outputs = [{'value': 0, 'script_pubkey': b''} for _ in range(vout + 1)]
        outputs[vout] = {
            'value': value,
            'script_pubkey': bytes.fromhex(script_pubkey_hex) if script_pubkey_hex else b''
        }
        self.analyzer.transaction_cache[prev_txid] = {'outputs': outputs}

    def test_legacy_sighash_vector(self):
        raw_tx = (
            "0200000001ad7f6cc26699d00283ef782d9fbe36b8310380c2a3fa457716d93b5746b483fc010000006a"
            "4730440220072bac42b585bbdb418d9f8a730bfc5575c3a29e4fef16979f67a22c08e2ecd902206efa623d"
            "c52a52521a67bc847c2d52f257c2accaaf62bef1b05b72a5a8179942012103786af4b32017ec640dba2d2a7"
            "e1fd5aa4a231a658e4cbc114d51c031576e19bcfdffffff02200fed380000000017a914dc8a8b615a69b4947"
            "91c2e30dc7d5022fc3a33ce871c483c5d130000001976a914cebb2851a9c7cfe2582c12ecaf7f3ff4383d1dc"
            "088ac00000000"
        )
        decoded = self.analyzer._decode_raw_transaction(raw_tx)
        input_index = 0
        prev_hash = decoded['inputs'][input_index]['prev_hash'][::-1].hex()
        prev_vout = decoded['inputs'][input_index]['prev_index']
        self._prepare_prevout(
            prev_hash,
            prev_vout,
            "76a914cebb2851a9c7cfe2582c12ecaf7f3ff4383d1dc088ac",
            value=84123728532,
        )

        expected_hash = int("48cd1ce4c9b8af7aa72c79dc853d85ce23cfbf2090383b84fb70947baed346c1", 16)
        sighash = self.analyzer._calculate_sighash(raw_tx, input_index)

        self.assertEqual(sighash, expected_hash)

    def test_bip143_p2sh_p2wpkh_vector(self):
        raw_tx = (
            "01000000000101db6b1b20aa0fd7b23880be2ecbd4a98130974cf4748fb66092ac4d3ceb1a547701000000"
            "1716001479091972186c449eb1ded22b78e40d009bdf0089feffffff02b8b4eb0b000000001976a914a457"
            "b684d7f0d539a46a45bbc043f35b59d0d96388ac0008af2f000000001976a914fd270b1ee6abcaea97fea7"
            "ad0402e8bd8ad6d77c88ac02473044022047ac8e878352d3ebbde1c94ce3a10d057c24175747116f8288e5"
            "d794d12d482f0220217f36a485cae903c713331d877c1f64677e3622ad4010726870540656fe9dcb012103"
            "ad1d8e89212f0b92c74d23bb710c00662ad1470198ac48c43f7d6f93a2a2687392040000"
        )
        decoded = self.analyzer._decode_raw_transaction(raw_tx)
        input_index = 0
        prev_hash = decoded['inputs'][input_index]['prev_hash'][::-1].hex()
        prev_vout = decoded['inputs'][input_index]['prev_index']
        self._prepare_prevout(
            prev_hash,
            prev_vout,
            "a9144733f37cf4db86fbc2efed2500b4f4e49f31202387",
            value=1000000000,
        )

        expected_hash = int("64f3b0f4dd2bb3aa1ce8566d220cc74dda9df97d8490cc81d89d735c92e59fb6", 16)
        sighash = self.analyzer._calculate_sighash(raw_tx, input_index)

        self.assertEqual(sighash, expected_hash)


if __name__ == "__main__":
    unittest.main()
