import pytest

from creatorcut.account_security import (
    PASSWORD_ALGORITHM,
    create_password_verifier,
    normalize_display_name,
    normalize_email,
    session_credentials,
    verify_password,
)


def test_password_verifier_is_salted_slow_and_constant_time_compatible():
    first = create_password_verifier("a sufficiently long password")
    second = create_password_verifier("a sufficiently long password")

    assert first["algorithm"] == PASSWORD_ALGORITHM
    assert first["iterations"] >= 600_000
    assert first["salt"] != second["salt"]
    assert first["digest"] != second["digest"]
    assert verify_password("a sufficiently long password", first)
    assert not verify_password("the wrong password", first)


def test_account_fields_are_normalized_and_bounded():
    assert normalize_email("  Creator@Example.COM ") == "creator@example.com"
    assert normalize_display_name("  My   Channel ") == "My Channel"
    with pytest.raises(ValueError, match="valid email"):
        normalize_email("not-an-email")
    with pytest.raises(ValueError, match="at least 12"):
        create_password_verifier("too short")


def test_session_credentials_store_only_a_digestable_opaque_token():
    session = session_credentials()

    assert session["token"] not in session["token_digest"]
    assert len(session["csrf_token"]) >= 32
    assert session["expires_at"] > session["created_at"]
