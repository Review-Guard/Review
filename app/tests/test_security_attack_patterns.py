import os
import sys
import unittest


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.app import create_app  # noqa: E402


class TestSecurityAttackPatterns(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_predict_handles_security_payloads_without_5xx(self):
        payloads = [
            "' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "${7*7}",
            "{{ config.items() }}",
            "<img src=x onerror=alert(1)>",
            "UNION SELECT password FROM users",
            "DROP TABLE reviews; --",
        ]
        for text in payloads:
            with self.subTest(text=text):
                res = self.client.post(
                    "/predict",
                    json={
                        "text": text,
                        "model_version": "v3",
                        "rating": 5,
                        "helpful_vote": 0,
                        "verified_purchase": 0,
                    },
                )
                self.assertIn(res.status_code, (200, 400, 413))
                self.assertNotEqual(res.status_code, 500)

    def test_predict_all_handles_security_payloads_without_5xx(self):
        payloads = [
            "' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "${7*7}",
            "{{ config.items() }}",
            "UNION SELECT * FROM users",
        ]
        for text in payloads:
            with self.subTest(text=text):
                res = self.client.post(
                    "/predict_all",
                    json={
                        "text": text,
                        "rating": 5,
                        "helpful_vote": 0,
                        "verified_purchase": 0,
                    },
                )
                self.assertIn(res.status_code, (200, 400, 413))
                self.assertNotEqual(res.status_code, 500)

    def test_predict_rejects_header_content_type_mismatch_safely(self):
        # Declared JSON but invalid body should fail safely with 400, not 500.
        res = self.client.post(
            "/predict",
            data="not-json-content",
            content_type="application/json",
            headers={"X-Forwarded-For": "127.0.0.1,10.0.0.1"},
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
