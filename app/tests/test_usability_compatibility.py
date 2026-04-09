import os
import re
import sys
import unittest


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.app import create_app  # noqa: E402


USER_AGENTS = {
    "chrome_windows": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "firefox_linux": (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
    ),
    "safari_macos": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    "mobile_ios": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
    ),
}


class TestUsabilityCompatibility(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_home_route_works_for_cross_browser_user_agents(self):
        for name, ua in USER_AGENTS.items():
            with self.subTest(user_agent=name):
                res = self.client.get("/", headers={"User-Agent": ua})
                self.assertEqual(res.status_code, 200)
                self.assertIn("text/html", res.content_type)

    def test_static_assets_load_for_cross_browser_user_agents(self):
        for name, ua in USER_AGENTS.items():
            with self.subTest(user_agent=name):
                css_res = self.client.get("/static/css/style.css", headers={"User-Agent": ua})
                js_res = self.client.get("/static/js/script.js", headers={"User-Agent": ua})
                self.assertEqual(css_res.status_code, 200)
                self.assertEqual(js_res.status_code, 200)

    def test_ui_contains_core_usability_elements(self):
        res = self.client.get("/")
        html = res.get_data(as_text=True)
        self.assertIn("id=\"reviewText\"", html)
        self.assertIn("id=\"analyzeBtn\"", html)
        self.assertIn("id=\"modelVersionSelect\"", html)
        self.assertIn("id=\"singleError\"", html)
        self.assertIn("Review Text", html)

    def test_model_selector_has_all_supported_versions(self):
        res = self.client.get("/")
        html = res.get_data(as_text=True)
        self.assertRegex(html, r"<option value=\"v1\">")
        self.assertRegex(html, r"<option value=\"v2\">")
        self.assertRegex(html, r"<option value=\"v3\".*selected")

    def test_predict_flow_consistent_for_user_agent_matrix(self):
        payload = {
            "text": "Fast shipping and authentic product.",
            "model_version": "v3",
            "rating": 5,
            "helpful_vote": 1,
            "verified_purchase": 1,
        }
        for name, ua in USER_AGENTS.items():
            with self.subTest(user_agent=name):
                res = self.client.post(
                    "/predict",
                    json=payload,
                    headers={"User-Agent": ua},
                )
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertIn("label", data)
                self.assertIn("fake_probability", data)

    def test_result_area_has_basic_accessibility_markers(self):
        res = self.client.get("/")
        html = res.get_data(as_text=True)
        # Basic semantic cues and visible labels for key output areas.
        self.assertIn("VERDICT", html)
        self.assertIn("FAKE CONFIDENCE", html)
        self.assertTrue(bool(re.search(r"id=\"resultCard\"", html)))


if __name__ == "__main__":
    unittest.main()
