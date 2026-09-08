"""Password, account-input, and opaque-session security primitives."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PASSWORD_ALGORITHM = "pbkdf2_hmac_sha256"
PASSWORD_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16
PASSWORD_DIGEST_BYTES = 32
MINIMUM_PASSWORD_CHARACTERS = 12
MAXIMUM_PASSWORD_BYTES = 1_024
SESSION_LIFETIME_DAYS = 7
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(value: Any) -> str:
    """Normalize and validate one login identifier."""
    if not isinstance(value, str):
        raise ValueError("Enter a valid email address")
    email = value.strip().casefold()
    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address")
    return email


def normalize_display_name(value: Any) -> str:
    """Validate a creator-facing profile name."""
    if not isinstance(value, str):
        raise ValueError("Enter a creator profile name")
    display_name = " ".join(value.strip().split())
    if not 1 <= len(display_name) <= 80:
        raise ValueError("Creator profile names must contain 1 to 80 characters")
    return display_name


def validate_password(value: Any) -> str:
    """Apply a length-first password policy and bound password-hash work."""
    if not isinstance(value, str):
        raise ValueError("Enter a password")
    encoded = value.encode("utf-8")
    if len(value) < MINIMUM_PASSWORD_CHARACTERS:
        raise ValueError(
            f"Passwords must contain at least {MINIMUM_PASSWORD_CHARACTERS} characters"
        )
    if len(encoded) > MAXIMUM_PASSWORD_BYTES:
        raise ValueError("Password is too long")
    return value


def password_digest(password: str, salt_hex: str, iterations: int) -> str:
    """Derive a slow salted password verifier using the standard library."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        iterations,
        dklen=PASSWORD_DIGEST_BYTES,
    ).hex()


DUMMY_PASSWORD_VERIFIER = {
    "algorithm": PASSWORD_ALGORITHM,
    "iterations": PASSWORD_ITERATIONS,
    "salt": "00" * PASSWORD_SALT_BYTES,
    "digest": password_digest(
        "creatorcut timing placeholder",
        "00" * PASSWORD_SALT_BYTES,
        PASSWORD_ITERATIONS,
    ),
}


def create_password_verifier(password: Any) -> dict[str, str | int]:
    """Create a versioned verifier suitable for persistent storage."""
    checked = validate_password(password)
    salt_hex = secrets.token_bytes(PASSWORD_SALT_BYTES).hex()
    return {
        "algorithm": PASSWORD_ALGORITHM,
        "iterations": PASSWORD_ITERATIONS,
        "salt": salt_hex,
        "digest": password_digest(checked, salt_hex, PASSWORD_ITERATIONS),
    }


def verify_password(password: Any, verifier: dict[str, Any]) -> bool:
    """Verify without leaking comparison timing or accepting unknown algorithms."""
    if not isinstance(password, str) or len(password.encode("utf-8")) > MAXIMUM_PASSWORD_BYTES:
        return False
    if verifier.get("algorithm") != PASSWORD_ALGORITHM:
        return False
    try:
        candidate = password_digest(
            password,
            str(verifier["salt"]),
            int(verifier["iterations"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return hmac.compare_digest(candidate, str(verifier.get("digest", "")))


def session_credentials() -> dict[str, str]:
    """Generate an opaque browser credential, CSRF token, and bounded expiry."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    return {
        "token": token,
        "token_digest": hashlib.sha256(token.encode()).hexdigest(),
        "csrf_token": secrets.token_urlsafe(32),
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=SESSION_LIFETIME_DAYS)).isoformat(),
    }


def session_token_digest(token: str) -> str:
    """Map a presented session token to the non-reversible database key."""
    return hashlib.sha256(token.encode()).hexdigest()


def main() -> None:
    """Create or promote an administrator without putting a password on the command line."""
    parser = argparse.ArgumentParser(description="Manage CreatorCut authenticated accounts")
    parser.add_argument(
        "--database", type=Path, default=Path("data/product/creatorcut.sqlite")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-admin", help="Create an administrator account")
    create.add_argument("--email", required=True)
    create.add_argument("--display-name", required=True)
    promote = subparsers.add_parser("promote-admin", help="Promote an existing account")
    promote.add_argument("--email", required=True)
    args = parser.parse_args()

    from creatorcut.product_store import ProductStore

    store = ProductStore(args.database)
    if args.command == "promote-admin":
        account = store.promote_account_to_admin(args.email)
        print(f"Administrator role granted to {account['email']}")
        return

    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        parser.error("Passwords do not match")
    account = store.register_account(
        args.display_name,
        args.email,
        password,
        is_admin=True,
    )
    print(f"Administrator account created for {account['email']}")


if __name__ == "__main__":
    main()
