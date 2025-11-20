import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from main import BitcoinBlockAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    return BitcoinBlockAnalyzer()


def test_private_key_to_public_key_generator_point(analyzer):
    private_key_hex = "0000000000000000000000000000000000000000000000000000000000000001"
    expected_public_key_hex = (
        "04"
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
    )
    public_key = analyzer._private_key_to_public_key(bytes.fromhex(private_key_hex))
    assert public_key is not None
    assert public_key.hex().upper() == expected_public_key_hex


def test_compress_public_key(analyzer):
    uncompressed_public_key = bytes.fromhex(
        "04"
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
    )
    compressed_public_key = analyzer._compress_public_key(uncompressed_public_key)
    assert compressed_public_key is not None
    assert compressed_public_key.hex().upper() == (
        "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    )


def test_generate_addresses_known_vector(analyzer):
    wif_key = "5HpHagT65TZzG1PH3CSu63k8DbpvD8s5ip4nEB3kEsreAnchuDf"
    addresses = analyzer._generate_addresses_from_wif(wif_key)
    assert addresses["legacy_uncompressed"] == "1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm"
    assert addresses["legacy_compressed"] == "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
    assert addresses["segwit"] == "3JvL6Ymt8MVWiCNHC7oWU6nLeHNJKLZGLN"
    assert addresses["bech32"] == "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
