"""Verify health, prediction, and metrics endpoints for a running API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def request_json(method: str, url: str, timeout: float, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--sample", type=Path, default=PROJECT_ROOT / "data" / "sample_request.json"
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    health = request_json("GET", f"{base_url}/health", args.timeout)
    prediction = request_json(
        "POST", f"{base_url}/predict", args.timeout, json=sample
    )
    metrics_response = requests.get(f"{base_url}/metrics", timeout=args.timeout)
    metrics_response.raise_for_status()

    assert health["status"] == "ok"
    assert prediction["prediction"] in (0, 1)
    assert 0 <= prediction["confidence"] <= 1
    assert "heart_api_requests_total" in metrics_response.text
    print(json.dumps({"health": health, "prediction": prediction}, indent=2))
    print("Smoke test passed: health, prediction, and metrics are available.")


if __name__ == "__main__":
    main()
