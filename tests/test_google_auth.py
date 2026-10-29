"""
Tests for "Sign in with Google": the signed session cookie (unit level) and
the HTTP behaviour of agent_server.py (real server on a free port, Google's
token check replaced by a fake -- no network, no real Google account).

Run from the project root:
    .venv/bin/python3 -m unittest discover -s tests -v
"""

import http.client
import json
import sys
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import google_auth

SECRET = "test-secret-not-used-anywhere-real"
ALLOWED = {"owner@example.com", "dev@example.com"}


class SessionCookieTests(unittest.TestCase):
    def test_round_trip(self):
        value = google_auth.make_session("Owner@Example.com", SECRET, 60, now=1000)
        self.assertEqual(google_auth.read_session(value, SECRET, now=1010), "owner@example.com")

    def test_expired(self):
        value = google_auth.make_session("owner@example.com", SECRET, 60, now=1000)
        self.assertIsNone(google_auth.read_session(value, SECRET, now=1060))

    def test_wrong_secret(self):
        value = google_auth.make_session("owner@example.com", SECRET, 60, now=1000)
        self.assertIsNone(google_auth.read_session(value, "another-secret", now=1010))

    def test_tampered_payload(self):
        value = google_auth.make_session("owner@example.com", SECRET, 60, now=1000)
        _, _, signature = value.partition(".")
        forged_payload = google_auth._b64(
            json.dumps({"e": "attacker@example.com", "x": 9999999999}).encode()
        )
        self.assertIsNone(
            google_auth.read_session(f"{forged_payload}.{signature}", SECRET, now=1010)
        )

    def test_garbage_never_raises(self):
        for junk in ["", "abc", ".", "a.b", "....", "é.é", None]:
            self.assertIsNone(google_auth.read_session(junk, SECRET))

    def test_empty_secret_never_validates(self):
        value = google_auth.make_session("owner@example.com", "", 60, now=1000)
        self.assertIsNone(google_auth.read_session(value, "", now=1010))


class CookieHeaderTests(unittest.TestCase):
    def test_finds_our_cookie_among_others(self):
        self.assertEqual(google_auth.cookie_value("a=1; shop_session=abc.def; b=2", "shop_session"), "abc.def")

    def test_missing_cookie(self):
        self.assertIsNone(google_auth.cookie_value("a=1", "shop_session"))
        self.assertIsNone(google_auth.cookie_value("", "shop_session"))
        self.assertIsNone(google_auth.cookie_value(None, "shop_session"))

    def test_google_g_state_cookie_does_not_hide_ours(self):
        # Regression: SimpleCookie dropped every cookie after this one.
        header = 'g_state={"i_l":0,"i_ll":1761700000000}; shop_session=abc.def'
        self.assertEqual(google_auth.cookie_value(header, "shop_session"), "abc.def")


class AllowlistTests(unittest.TestCase):
    def test_allowed_is_case_and_space_insensitive(self):
        self.assertTrue(google_auth.email_allowed(" OWNER@example.com ", ALLOWED))

    def test_not_allowed(self):
        self.assertFalse(google_auth.email_allowed("x@example.com", ALLOWED))
        self.assertFalse(google_auth.email_allowed("", ALLOWED))
        self.assertFalse(google_auth.email_allowed(None, ALLOWED))


class CredentialPrecheckTests(unittest.TestCase):
    def test_malformed_credentials_rejected_before_any_network_call(self):
        for bad in [None, "", "nodots", "a.b", "a.b.c.d", "a.b." + "c" * 5000, 123]:
            with self.assertRaises(ValueError):
                google_auth.verify_google_credential(bad, "client-id")


class ServerHarness(unittest.TestCase):
    """Runs the real Handler on a free port; defines no tests of its own."""

    TOKEN = "test-bearer-token"
    GOOGLE_ON = True

    @classmethod
    def setUpClass(cls):
        import agent_server

        cls.agent_server = agent_server
        cls.patches = [
            mock.patch.object(agent_server, "AGENT_API_TOKEN", cls.TOKEN),
            mock.patch.object(agent_server, "GOOGLE_AUTH_ENABLED", cls.GOOGLE_ON),
            mock.patch.object(agent_server, "GOOGLE_CLIENT_ID", "client-id.apps.googleusercontent.com"),
            mock.patch.object(agent_server, "ALLOWED_GOOGLE_EMAILS", ALLOWED),
            mock.patch.object(agent_server, "SESSION_SECRET", SECRET),
            mock.patch.object(agent_server.monitoring, "record_auth_failure"),
        ]
        for p in cls.patches:
            p.start()

        cls.server = HTTPServer(("127.0.0.1", 0), agent_server.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        for p in cls.patches:
            p.stop()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response, json.loads(raw) if raw else None

    def sign_in(self, email="owner@example.com", forwarded_https=False):
        """Signs in through /auth/google with a faked Google check; returns the cookie header value."""
        headers = {"Content-Type": "application/json"}
        if forwarded_https:
            headers["X-Forwarded-Proto"] = "https"

        with mock.patch.object(
            self.agent_server, "verify_google_credential", return_value=email
        ):
            response, _ = self.request("POST", "/auth/google", {"credential": "x.y.z"}, headers)

        cookie = response.getheader("Set-Cookie")
        return response, cookie


class ServerAuthTests(ServerHarness):
    def test_config_is_public_and_exposes_only_the_client_id(self):
        response, data = self.request("GET", "/auth/config")
        self.assertEqual(response.status, 200)
        self.assertEqual(data, {"google_client_id": "client-id.apps.googleusercontent.com"})

    def test_me_when_signed_out(self):
        response, data = self.request("GET", "/auth/me")
        self.assertEqual(response.status, 200)
        self.assertEqual(data, {"signed_in": False, "email": None})

    def test_data_endpoints_still_require_auth(self):
        response, _ = self.request("GET", "/health")
        self.assertEqual(response.status, 401)
        response, _ = self.request("GET", "/inventory")
        self.assertEqual(response.status, 401)

    def test_bearer_token_still_works(self):
        response, data = self.request("GET", "/health", headers={"Authorization": f"Bearer {self.TOKEN}"})
        self.assertEqual(response.status, 200)
        self.assertEqual(data["status"], "ok")

    def test_wrong_bearer_token_rejected(self):
        response, _ = self.request("GET", "/health", headers={"Authorization": "Bearer nope"})
        self.assertEqual(response.status, 401)

    def test_sign_in_sets_a_hardened_cookie_and_it_authenticates(self):
        response, cookie = self.sign_in(forwarded_https=True)
        self.assertEqual(response.status, 200)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        self.assertIn("Secure", cookie)

        session = cookie.split(";")[0]
        response, data = self.request("GET", "/health", headers={"Cookie": session})
        self.assertEqual(response.status, 200)

        response, data = self.request("GET", "/auth/me", headers={"Cookie": session})
        self.assertEqual(data, {"signed_in": True, "email": "owner@example.com"})

    def test_session_works_alongside_googles_g_state_cookie(self):
        _, cookie = self.sign_in()
        session = cookie.split(";")[0]
        header = 'g_state={"i_l":0,"i_ll":1761700000000}; ' + session

        response, _ = self.request("GET", "/health", headers={"Cookie": header})
        self.assertEqual(response.status, 200)

    def test_cookie_not_marked_secure_over_plain_http(self):
        _, cookie = self.sign_in(forwarded_https=False)
        self.assertNotIn("Secure", cookie)

    def test_cookie_write_needs_the_csrf_header(self):
        _, cookie = self.sign_in()
        session = cookie.split(";")[0]

        response, _ = self.request("DELETE", "/history", headers={"Cookie": session})
        self.assertEqual(response.status, 401)

        response, _ = self.request(
            "DELETE", "/history", headers={"Cookie": session, "X-Requested-With": "shop-ui"}
        )
        self.assertEqual(response.status, 200)

    def test_bearer_write_needs_no_csrf_header(self):
        response, _ = self.request(
            "DELETE", "/history", headers={"Authorization": f"Bearer {self.TOKEN}"}
        )
        self.assertEqual(response.status, 200)

    def test_email_not_on_allowlist_gets_403_and_no_cookie(self):
        response, cookie = self.sign_in(email="stranger@example.com")
        self.assertEqual(response.status, 403)
        self.assertIsNone(cookie)

    def test_removed_from_allowlist_locks_out_existing_cookie(self):
        _, cookie = self.sign_in(email="dev@example.com")
        session = cookie.split(";")[0]

        response, _ = self.request("GET", "/health", headers={"Cookie": session})
        self.assertEqual(response.status, 200)

        with mock.patch.object(self.agent_server, "ALLOWED_GOOGLE_EMAILS", {"owner@example.com"}):
            response, _ = self.request("GET", "/health", headers={"Cookie": session})
            self.assertEqual(response.status, 401)

    def test_invalid_google_credential_gets_401(self):
        with mock.patch.object(
            self.agent_server, "verify_google_credential", side_effect=ValueError("bad")
        ):
            response, _ = self.request(
                "POST", "/auth/google", {"credential": "x.y.z"}, {"Content-Type": "application/json"}
            )
        self.assertEqual(response.status, 401)

    def test_google_unreachable_gets_502_not_401(self):
        with mock.patch.object(
            self.agent_server, "verify_google_credential", side_effect=OSError("network")
        ):
            response, _ = self.request(
                "POST", "/auth/google", {"credential": "x.y.z"}, {"Content-Type": "application/json"}
            )
        self.assertEqual(response.status, 502)

    def test_forged_cookie_rejected(self):
        forged = google_auth.make_session("owner@example.com", "wrong-secret", 3600)
        response, _ = self.request("GET", "/health", headers={"Cookie": f"shop_session={forged}"})
        self.assertEqual(response.status, 401)

    def test_logout_clears_cookie(self):
        response, _ = self.request("POST", "/auth/logout")
        self.assertEqual(response.status, 200)
        self.assertIn("Max-Age=0", response.getheader("Set-Cookie"))


class GoogleDisabledTests(ServerHarness):
    """With Google sign-in unconfigured, everything must fail closed."""

    GOOGLE_ON = False

    def test_config_exposes_no_client_id(self):
        _, data = self.request("GET", "/auth/config")
        self.assertEqual(data, {"google_client_id": None})

    def test_google_endpoint_refuses(self):
        response, _ = self.request("POST", "/auth/google", {"credential": "x.y.z"})
        self.assertEqual(response.status, 503)

    def test_session_cookie_is_ignored(self):
        valid = google_auth.make_session("owner@example.com", SECRET, 3600)
        response, _ = self.request("GET", "/health", headers={"Cookie": f"shop_session={valid}"})
        self.assertEqual(response.status, 401)


if __name__ == "__main__":
    unittest.main()
