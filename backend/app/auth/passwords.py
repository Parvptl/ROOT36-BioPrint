"""Password hashing.

Argon2id, via argon2-cffi. Chosen over bcrypt/PBKDF2 because it is memory-hard:
an attacker with a leaked `users` table cannot trade cheap parallel GPU cores
for speed the way they can against a CPU-only KDF.

What this mitigates: offline cracking of the password column after a database
compromise. What it does NOT mitigate: a weak password that falls to a
dictionary attack regardless of KDF cost, phishing, or credential reuse. Those
are exactly the cases BioPrint's behavioural layer exists to catch, since the
attacker there holds a *correct* password.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHash

# Parameters sized for an interactive login on commodity hardware: ~64 MiB and
# a few tens of milliseconds. High enough to be expensive in bulk, low enough
# that it is not the dominant term in our authentication latency budget.
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=64 * 1024,  # KiB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

# Precomputed hash of a value no user can supply. Verifying against this lets a
# login for a non-existent username burn the same CPU time as a real one, so
# response timing does not reveal which usernames are registered.
_DUMMY_HASH = _hasher.hash("bioprint-nonexistent-account-placeholder")


def hash_password(password: str) -> str:
    """Return an Argon2id encoded hash. The plaintext is never persisted."""
    return _hasher.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    """Constant-effort verification. Returns False instead of raising."""
    try:
        _hasher.verify(encoded_hash, password)
        return True
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False


def waste_time_like_a_real_verify() -> None:
    """Spend the same work as verify_password for an unknown username.

    Called on the 'user not found' path so that username enumeration by
    response-time measurement is not free.
    """
    try:
        _hasher.verify(_DUMMY_HASH, "not-the-password")
    except (VerifyMismatchError, VerificationError, InvalidHash):
        pass


def needs_rehash(encoded_hash: str) -> bool:
    """True when a stored hash predates the current cost parameters."""
    try:
        return _hasher.check_needs_rehash(encoded_hash)
    except InvalidHash:
        return True
