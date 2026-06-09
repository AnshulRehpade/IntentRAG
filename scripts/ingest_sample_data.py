"""
Ingest sample documents into IntentRAG.

Reads all .txt files from data/{intent_category}/ and ingests them
into the appropriate Qdrant collections via the API.

Usage:
    # Start Docker services first:
    docker compose up -d

    # Start the API server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python scripts/ingest_sample_data.py

    # Or specify a custom base URL and credentials:
    python scripts/ingest_sample_data.py --url http://localhost:8000 --email admin@demo.com --password password123
"""

import argparse
import os
import sys
from pathlib import Path

import httpx

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INTENT_CATEGORIES = ["factual", "person", "time", "location", "explanation", "other"]


def get_token(base_url: str, email: str, password: str) -> str:
    """Login and get JWT token."""
    resp = httpx.post(
        f"{base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=10.0,
    )
    data = resp.json()

    if not data.get("success"):
        # Try registering if login fails
        print(f"  Login failed, attempting registration...")
        resp = httpx.post(
            f"{base_url}/auth/register",
            json={
                "email": email,
                "password": password,
                "tenant_name": "demo",
                "role": "admin",
            },
            timeout=10.0,
        )
        data = resp.json()
        if not data.get("success"):
            print(f"  ❌ Registration also failed: {data.get('error')}")
            sys.exit(1)

    token = data["data"]["token"]
    tenant = data["data"].get("tenant_name", "demo")
    role = data["data"].get("role", "unknown")
    print(f"  ✅ Authenticated as {email} (role: {role}, tenant: {tenant})")
    return token


def ingest_file(base_url: str, token: str, filepath: Path, intent_category: str) -> dict:
    """Ingest a single file via the /ingest endpoint."""
    with open(filepath, "rb") as f:
        resp = httpx.post(
            f"{base_url}/ingest",
            headers={"Authorization": f"Bearer {token}"},
            data={"intent_category": intent_category, "source": filepath.name},
            files={"file": (filepath.name, f, "text/plain")},
            timeout=60.0,
        )
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Ingest sample documents into IntentRAG")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--email", default="admin@demo.com", help="Admin email")
    parser.add_argument("--password", default="password123", help="Admin password")
    parser.add_argument("--category", default=None, help="Ingest only this category")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("=" * 60)
    print("IntentRAG — Sample Data Ingestion")
    print("=" * 60)
    print()

    # Check API is running
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5.0)
        health = resp.json()
        print(f"API Status: {health['data']['status']}")
        print(f"Services: {health['data']['services']}")
    except Exception as e:
        print(f"❌ Cannot connect to API at {base_url}: {e}")
        print("   Make sure the server is running: uvicorn app.main:app --port 8000")
        sys.exit(1)

    print()

    # Authenticate
    print("Authenticating...")
    token = get_token(base_url, args.email, args.password)
    print()

    # Discover and ingest files
    categories = [args.category] if args.category else INTENT_CATEGORIES
    total_files = 0
    total_chunks = 0
    errors = []

    for category in categories:
        category_dir = DATA_DIR / category
        if not category_dir.exists():
            print(f"⚠️  No data directory for '{category}' — skipping")
            continue

        files = sorted(category_dir.glob("*.txt"))
        if not files:
            print(f"⚠️  No .txt files in data/{category}/ — skipping")
            continue

        print(f"📁 [{category}] — {len(files)} file(s)")

        for filepath in files:
            result = ingest_file(base_url, token, filepath, category)

            if result.get("success"):
                chunks = result["data"]["chunks_created"]
                total_chunks += chunks
                total_files += 1
                print(f"   ✅ {filepath.name} → {chunks} chunks")
            else:
                error_msg = result.get("error", "Unknown error")
                errors.append(f"{category}/{filepath.name}: {error_msg}")
                print(f"   ❌ {filepath.name}: {error_msg}")

        print()

    # Summary
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Files ingested: {total_files}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Errors: {len(errors)}")
    if errors:
        print()
        print("  Errors:")
        for err in errors:
            print(f"    - {err}")
    print()
    print("Done! You can now query the system:")
    print(f'  curl -X POST {base_url}/query \\')
    print(f'    -H "Authorization: Bearer <token>" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"query": "What is machine learning?"}}\'')


if __name__ == "__main__":
    main()
