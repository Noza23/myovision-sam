import hashlib


def hash_bytes(bb: bytes) -> str:
    """Hash a numpy array for caching."""
    hash = hashlib.sha256(bb)
    return hash.hexdigest()
