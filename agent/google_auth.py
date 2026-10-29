"""
google_auth.py

"Sign in with Google" plus a signed session cookie, for the web UI.

Flow: the browser shows Google's sign-in button and gets back an ID token
(a signed JWT from Google). The server verifies that token with Google's
public keys, checks the email against an explicit allowlist, and then hands
the browser its own session cookie. From then on the cookie authenticates
requests -- Google isn't contacted again until the session expires.

Why a hand-rolled signed cookie instead of a server-side session store: one
shop, one or two users, and a server that restarts (deploys, health-check
restarts). A stateless HMAC-signed value survives restarts with no storage.
The one thing this gives up is instant remote logout of a single session;
removing an email from the allowlist still takes effect immediately, since
the allowlist is re-checked on every request, not just at sign-in.
"""

import base64
import hashlib
import hmac
import json
import time

SESSION_COOKIE = "shop_session"

# A Google ID token is a JWT of roughly 1-2 KB. Anything wildly outside that
# is junk -- reject it before spending a network round trip to Google on it
# (this endpoint is reachable without being signed in).
MAX_CREDENTIAL_LENGTH = 4096


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def cookie_value(cookie_header, name):
    """
    Pulls one cookie out of a Cookie header, ignoring all the others.

    Deliberately not http.cookies.SimpleCookie: it discards EVERY cookie
    once it hits one it can't parse, and Google's own sign-in script sets
    a `g_state={"i_l":0}` cookie on this site that it can't. That silently
    dropped our session cookie, so a successful sign-in looked signed-out.
    """
    for part in (cookie_header or "").split(";"):
        key, _, value = part.strip().partition("=")

        if key == name:
            return value

    return None


def _sign(secret, payload):
    return _b64(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())


def make_session(email, secret, ttl_seconds, now=None):
    """Returns the cookie value: base64(payload).signature."""
    expires_at = int((now if now is not None else time.time()) + ttl_seconds)
    payload = _b64(json.dumps({"e": email.lower(), "x": expires_at}).encode("utf-8"))
    return f"{payload}.{_sign(secret, payload)}"


def read_session(value, secret, now=None):
    """
    Returns the signed-in email, or None for anything wrong with the cookie:
    malformed, tampered with, signed with a different secret, or expired.
    Never raises on bad input -- a bad cookie is just "not signed in".
    """
    # isascii: signing encodes as ASCII and compare_digest rejects non-ASCII
    # str, so a hostile cookie with e.g. an accented character would raise
    # instead of just meaning "not signed in".
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not secret
        or "." not in value
    ):
        return None

    payload, _, signature = value.partition(".")

    # compare_digest: don't leak how many leading characters of the
    # signature were right via response timing.
    if not hmac.compare_digest(signature, _sign(secret, payload)):
        return None

    try:
        data = json.loads(_unb64(payload))
        email = data["e"]
        expires_at = data["x"]
    except (ValueError, KeyError, TypeError):
        return None

    if not isinstance(email, str) or not isinstance(expires_at, (int, float)):
        return None

    if (now if now is not None else time.time()) >= expires_at:
        return None

    return email


def email_allowed(email, allowed_emails):
    return bool(email) and email.strip().lower() in allowed_emails


def verify_google_credential(credential, client_id):
    """
    Verifies a Google ID token (signature, expiry, issuer, and that it was
    issued for THIS app's client ID) and returns the verified, lower-cased
    email. Raises ValueError for anything not acceptable.

    The `google-auth` library is imported here, lazily, so the rest of the
    server (and the local dev setup without Google sign-in configured)
    doesn't need it installed just to start.
    """
    if (
        not isinstance(credential, str)
        or credential.count(".") != 2
        or len(credential) > MAX_CREDENTIAL_LENGTH
    ):
        raise ValueError("malformed credential")

    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    claims = id_token.verify_oauth2_token(
        credential, google_requests.Request(), client_id
    )

    # An address the user hasn't verified with Google isn't evidence of
    # anything -- only trust verified ones.
    if claims.get("email_verified") is not True:
        raise ValueError("Google reports this email as unverified")

    email = (claims.get("email") or "").strip().lower()

    if not email:
        raise ValueError("no email in credential")

    return email
