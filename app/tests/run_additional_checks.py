"""Run additional non-unit checks for submission reporting.

Covers:
- Integration/API contract checks
- Input-validation and security-oriented negative checks
- Lightweight local performance check (sequential requests)
"""

from __future__ import annotations

import json
import os
import sys
import time
from statistics import mean

try:
    from sklearn.exceptions import InconsistentVersionWarning
    import warnings

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PHASE1_DIR not in sys.path:
    sys.path.insert(0, PHASE1_DIR)

from backend.app import create_app  # noqa: E402


def main() -> int:
    app = create_app()
    client = app.test_client()
    results: list[dict[str, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append(
            {
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
            }
        )

    # Functional/API checks
    res = client.get("/health")
    body = res.get_json() if res.is_json else {}
    record(
        "IT-01 health endpoint",
        res.status_code == 200 and body.get("status") == "ok",
        f"status={res.status_code}, body={body}",
    )

    valid_payload = {
        "text": "Great product, fast delivery and authentic quality.",
        "model_version": "v3",
        "rating": 5,
        "helpful_vote": 2,
        "verified_purchase": 1,
    }
    res = client.post("/predict", json=valid_payload)
    body = res.get_json() if res.is_json else {}
    record(
        "IT-02 predict valid payload",
        res.status_code == 200 and "label" in body and "fake_probability" in body,
        f"status={res.status_code}, keys={list(body.keys()) if isinstance(body, dict) else 'n/a'}",
    )

    # Security/validation checks
    res = client.post("/predict", data="not-json", content_type="text/plain")
    record("SEC-01 reject non-JSON", res.status_code == 400, f"status={res.status_code}")

    res = client.post("/predict", json={"rating": 3})
    record("SEC-02 reject missing text", res.status_code == 400, f"status={res.status_code}")

    res = client.post("/predict", json={"text": "   "})
    record("SEC-03 reject empty text", res.status_code == 400, f"status={res.status_code}")

    res = client.post("/predict", json={"text": "a" * 10001})
    record("SEC-04 reject oversized text", res.status_code == 413, f"status={res.status_code}")

    res = client.post("/predict", json={"text": "<script>alert(1)</script>", "rating": "abc"})
    record("SEC-05 reject invalid numeric types", res.status_code == 400, f"status={res.status_code}")

    # Integration comparison endpoint
    res = client.post(
        "/predict_all",
        json={
            "text": "This review looks suspiciously perfect.",
            "rating": 5,
            "helpful_vote": 0,
            "verified_purchase": 0,
        },
    )
    body = res.get_json() if res.is_json else {}
    record(
        "IT-03 predict_all contract",
        res.status_code == 200 and isinstance(body.get("predictions"), dict),
        f"status={res.status_code}, keys={list(body.keys()) if isinstance(body, dict) else 'n/a'}",
    )

    # Lightweight performance check (local sequential requests)
    latencies = []
    for _ in range(30):
        start = time.perf_counter()
        res = client.post("/predict", json=valid_payload)
        latencies.append((time.perf_counter() - start) * 1000)
        if res.status_code != 200:
            record("PERF-01 predict stability (30 requests)", False, f"status={res.status_code}")
            break
    else:
        p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1]
        record(
            "PERF-01 predict stability (30 requests)",
            True,
            f"mean_ms={mean(latencies):.2f}, p95_ms={p95:.2f}, max_ms={max(latencies):.2f}",
        )

    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = sum(1 for item in results if item["status"] == "FAIL")

    print(
        json.dumps(
            {
                "summary": {
                    "total": len(results),
                    "passed": passed,
                    "failed": failed,
                    "pass_rate": round((passed / len(results)) * 100, 2) if results else 0.0,
                },
                "results": results,
            },
            indent=2,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
