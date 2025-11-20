import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crypto_utils import SecureCryptoUtils, ECDSA_AVAILABLE

try:
    from ecdsa.util import sigencode_der
except Exception:  # pragma: no cover - handled by pytest skip when ecdsa missing
    sigencode_der = None

pytestmark = pytest.mark.skipif(not ECDSA_AVAILABLE, reason="ecdsa library is required for crypto tests")


def _generate_private_key_int():
    from ecdsa import SigningKey, SECP256k1

    signing_key = SigningKey.generate(curve=SECP256k1)
    return int.from_bytes(signing_key.to_string(), "big"), signing_key


def test_public_key_to_coordinates_handles_compressed_key():
    private_key_int, signing_key = _generate_private_key_int()
    utils = SecureCryptoUtils()

    public_key_hex = utils.private_key_to_public_key(private_key_int)
    coords = utils.public_key_to_coordinates(public_key_hex)

    assert coords is not None
    point = signing_key.get_verifying_key().pubkey.point
    assert coords == (point.x(), point.y())


def test_public_key_to_coordinates_accepts_uncompressed_hex():
    _, signing_key = _generate_private_key_int()
    utils = SecureCryptoUtils()

    uncompressed_hex = signing_key.get_verifying_key().to_string("uncompressed").hex()
    coords = utils.public_key_to_coordinates(uncompressed_hex)

    assert coords is not None
    point = signing_key.get_verifying_key().pubkey.point
    assert coords == (point.x(), point.y())


def test_private_key_to_public_key_supports_uncompressed_output():
    private_key_int, signing_key = _generate_private_key_int()
    utils = SecureCryptoUtils()

    uncompressed_hex = utils.private_key_to_public_key(private_key_int, compressed=False)

    assert len(uncompressed_hex) == 130
    point = signing_key.get_verifying_key().pubkey.point
    recovered = utils.public_key_to_coordinates(uncompressed_hex)
    assert recovered == (point.x(), point.y())


def test_is_valid_public_key_accepts_multiple_encodings():
    private_key_int, signing_key = _generate_private_key_int()
    utils = SecureCryptoUtils()

    compressed_hex = utils.private_key_to_public_key(private_key_int, compressed=True)
    raw_hex = signing_key.get_verifying_key().to_string().hex()
    uncompressed_hex = signing_key.get_verifying_key().to_string("uncompressed").hex()

    assert utils.is_valid_public_key(compressed_hex)
    assert utils.is_valid_public_key(raw_hex)
    assert utils.is_valid_public_key(uncompressed_hex)


def test_verify_signature_handles_compressed_public_keys():
    private_key_int, signing_key = _generate_private_key_int()
    utils = SecureCryptoUtils()

    message_hash = b"\x42" * 32
    signature = signing_key.sign_digest_deterministic(message_hash, sigencode=sigencode_der)

    compressed_hex = utils.private_key_to_public_key(private_key_int, compressed=True)

    assert utils.verify_signature(compressed_hex, signature.hex(), message_hash)
