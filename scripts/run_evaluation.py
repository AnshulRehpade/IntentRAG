"""
Run RAGAS evaluation against the sample test set.

Usage:
    # Make sure the API is running and data is ingested:
    python scripts/ingest_sample_data.py

    # Then run evaluation:
    python scripts/run_evaluation.py

    # Or with custom options:
    python scripts/run_evaluation.py --url http://localhost:8000 --email admin@demo.com --password password123
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
TEST_SET_PATH = PROJECT_ROOT / "data" / "eval_test_set.json"


def get_token(base_url: str, email: str, password: str) -> str:
    """Login and get JWT token."""
    resp = httpx.post(
        f"{base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=10.0,
    )
    data = resp.json()
    if not data.get("success"):
        print(f"❌ Login failed: {data.get('error')}")
        sys.exit(1)
    return data["data"]["token"]


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on IntentRAG")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--email", default="admin@demo.com", help="Admin email")
    parser.add_argument("--password", default="password123", help="Admin password")
    parser.add_argument("--test-set", default=str(TEST_SET_PATH), help="Path to test set JSON")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("=" * 60)
    print("IntentRAG — RAGAS Evaluation")
    print("=" * 60)
    print()

    # Load test set
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        print(f"❌ Test set not found: {test_set_path}")
        sys.exit(1)

    with open(test_set_path) as f:
        test_set = json.load(f)

    print(f"Test set: {test_set_path.name} ({len(test_set)} items)")
    print()

    # Authenticate
    print("Authenticating...")
    token = get_token(base_url, args.email, args.password)
    print("  ✅ Authenticated")
    print()

    # Run evaluation
    print(f"Running evaluation on {len(test_set)} items...")
    print("(This may take a few minutes — each item runs through the full pipeline)")
    print()

    resp = httpx.post(
        f"{base_url}/eval",
        headers={"Authorization": f"Bearer {token}"},
        json={"test_set": test_set},
        timeout=300.0,  # 5 min timeout for full eval
    )
    data = resp.json()

    if not data.get("success"):
        print(f"❌ Evaluation failed: {data.get('error')}")
        sys.exit(1)

    result = data["data"]

    # Print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    print(f"Items submitted: {result['num_items_submitted']}")
    print(f"Items evaluated: {result['num_items_evaluated']}")
    print(f"Latency: {result['latency_ms']}ms")
    print()

    if result["aggregate_scores"]:
        print("📊 AGGREGATE SCORES")
        print("-" * 40)
        for metric, score in result["aggregate_scores"].items():
            if score is not None:
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {metric:25s} {bar} {score:.4f}")
            else:
                print(f"  {metric:25s} N/A")
        print()

    if result["per_item_scores"]:
        print("📝 PER-ITEM RESULTS")
        print("-" * 40)
        for item in result["per_item_scores"]:
            print(f"  [{item.get('intent', '?'):12s}] {item['question'][:50]}")
            if "scores" in item:
                scores_str = " | ".join(
                    f"{k}={v:.2f}" for k, v in item["scores"].items() if v is not None
                )
                print(f"               {scores_str}")
        print()

    if result["errors"]:
        print(f"⚠️  ERRORS ({len(result['errors'])})")
        print("-" * 40)
        for err in result["errors"]:
            print(f"  - {err}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
