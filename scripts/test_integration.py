"""
Full Integration Test — Multi-Tenant Isolation

This script verifies the complete IntentRAG pipeline with two tenants:
1. Registers two separate tenants (acme, globex)
2. Ingests different documents into each tenant
3. Verifies tenant isolation (acme can't see globex's data)
4. Runs queries and validates the full pipeline
5. Tests role-based access control

Prerequisites:
    docker compose up -d
    uvicorn app.main:app --port 8000

Usage:
    python scripts/test_integration.py
    python scripts/test_integration.py --url http://localhost:8000
"""

import argparse
import io
import sys
import time

import httpx

# ----- Test Data -----

ACME_DOCS = {
    "factual": (
        "ACME Corp was founded in 2015 and specializes in cloud infrastructure. "
        "Their flagship product is AcmeCloud, a Kubernetes management platform. "
        "ACME has 500 employees and is headquartered in San Francisco. "
        "Their annual revenue is $50 million as of 2024."
    ),
    "person": (
        "Jane Smith is the CEO of ACME Corp. She joined in 2018 after working at Google for 10 years. "
        "Bob Johnson is the CTO. He previously worked at Amazon Web Services. "
        "Alice Chen is the VP of Engineering with 15 years of experience in distributed systems."
    ),
}

GLOBEX_DOCS = {
    "factual": (
        "Globex Industries is a biotech company founded in 2010. "
        "They focus on gene therapy and personalized medicine. "
        "Globex has 200 employees and is based in Boston, Massachusetts. "
        "Their lead product GX-100 is in Phase 3 clinical trials."
    ),
    "person": (
        "Dr. Michael Lee is the CEO of Globex Industries. He holds a PhD in molecular biology from MIT. "
        "Sarah Park is the Chief Science Officer. She discovered the GX-100 compound in 2018. "
        "David Kim is the VP of Clinical Operations."
    ),
}


class IntegrationTest:
    """Runs the full integration test suite."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self):
        """Run all tests."""
        print("=" * 70)
        print("IntentRAG — Full Integration Test (Multi-Tenant)")
        print("=" * 70)
        print()

        # Check API
        self._check_health()
        print()

        # Register tenants
        acme_admin = self._register_tenant("acme", "admin@acme.com", "acmepass123")
        globex_admin = self._register_tenant("globex", "admin@globex.com", "globexpass123")
        print()

        # Test role-based access
        self._test_roles(acme_admin)
        print()

        # Ingest tenant-specific documents
        self._ingest_docs("acme", acme_admin, ACME_DOCS)
        self._ingest_docs("globex", globex_admin, GLOBEX_DOCS)
        print()

        # Test tenant isolation on queries
        self._test_tenant_isolation(acme_admin, globex_admin)
        print()

        # Test query pipeline
        self._test_query_pipeline(acme_admin, "acme")
        self._test_query_pipeline(globex_admin, "globex")
        print()

        # Test auth/me endpoint
        self._test_me_endpoint(acme_admin, "acme")
        print()

        # Summary
        self._print_summary()

    def _check_health(self):
        """Verify API is running."""
        print("🔍 Checking API health...")
        try:
            resp = self.client.get(f"{self.base_url}/health")
            data = resp.json()
            services = data["data"]["services"]

            self._assert("API is up", services["api"] == "up")
            self._assert("PostgreSQL is up", services["postgres"] == "up")
            self._assert("Qdrant is up", services["qdrant"] == "up")

        except httpx.ConnectError:
            print("❌ Cannot connect to API. Make sure the server is running.")
            sys.exit(1)

    def _register_tenant(self, tenant_name: str, email: str, password: str) -> dict:
        """Register a tenant and return auth info."""
        print(f"📝 Registering tenant: {tenant_name}")

        resp = self.client.post(
            f"{self.base_url}/auth/register",
            json={"email": email, "password": password, "tenant_name": tenant_name},
        )
        data = resp.json()

        if data.get("success"):
            token = data["data"]["token"]
            tenant_id = data["data"]["tenant_id"]
            role = data["data"]["role"]
            self._assert(f"  {tenant_name} registered", True)
            self._assert(f"  First user is admin", role == "admin")
            return {"token": token, "tenant_id": tenant_id, "tenant_name": tenant_name}
        else:
            # Might already exist — try login
            resp = self.client.post(
                f"{self.base_url}/auth/login",
                json={"email": email, "password": password},
            )
            data = resp.json()
            if data.get("success"):
                self._assert(f"  {tenant_name} logged in (already existed)", True)
                return {
                    "token": data["data"]["token"],
                    "tenant_id": data["data"]["tenant_id"],
                    "tenant_name": tenant_name,
                }
            else:
                self._assert(f"  {tenant_name} auth failed: {data.get('error')}", False)
                return {"token": "", "tenant_id": "", "tenant_name": tenant_name}

    def _test_roles(self, admin_auth: dict):
        """Test role-based access control."""
        print("🔒 Testing role-based access...")

        token = admin_auth["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Register a reader in the same tenant
        resp = self.client.post(
            f"{self.base_url}/auth/register",
            json={
                "email": f"reader@{admin_auth['tenant_name']}.com",
                "password": "readerpass",
                "tenant_name": admin_auth["tenant_name"],
                "role": "reader",
            },
        )
        data = resp.json()

        if data.get("success"):
            reader_token = data["data"]["token"]
            reader_headers = {"Authorization": f"Bearer {reader_token}"}

            # Reader can query
            resp = self.client.post(
                f"{self.base_url}/query",
                json={"query": "test"},
                headers=reader_headers,
            )
            self._assert("  Reader can access /query", resp.status_code != 403)

            # Reader cannot ingest
            resp = self.client.post(
                f"{self.base_url}/ingest",
                data={"intent_category": "factual"},
                files={"file": ("t.txt", b"test", "text/plain")},
                headers=reader_headers,
            )
            self._assert("  Reader blocked from /ingest", resp.status_code == 403)

            # Reader cannot eval
            resp = self.client.post(
                f"{self.base_url}/eval",
                json={"test_set": [{"question": "q", "ground_truth": "a"}]},
                headers=reader_headers,
            )
            self._assert("  Reader blocked from /eval", resp.status_code == 403)

        # Admin can do everything
        resp = self.client.post(
            f"{self.base_url}/eval",
            json={"test_set": [{"question": "q", "ground_truth": "a"}]},
            headers=headers,
        )
        self._assert("  Admin can access /eval", resp.status_code != 403)

    def _ingest_docs(self, tenant_name: str, auth: dict, docs: dict):
        """Ingest documents for a tenant."""
        print(f"📄 Ingesting documents for {tenant_name}...")

        headers = {"Authorization": f"Bearer {auth['token']}"}

        for intent, content in docs.items():
            resp = self.client.post(
                f"{self.base_url}/ingest",
                data={"intent_category": intent, "source": f"{tenant_name}_{intent}.txt"},
                files={"file": (f"{tenant_name}_{intent}.txt", content.encode(), "text/plain")},
                headers=headers,
            )
            data = resp.json()
            if data.get("success"):
                chunks = data["data"]["chunks_created"]
                self._assert(f"  [{intent}] ingested ({chunks} chunks)", True)
            else:
                self._assert(f"  [{intent}] ingest failed: {data.get('error')}", False)

    def _test_tenant_isolation(self, acme_auth: dict, globex_auth: dict):
        """Verify that tenants cannot see each other's data."""
        print("🔐 Testing tenant isolation...")

        acme_headers = {"Authorization": f"Bearer {acme_auth['token']}"}
        globex_headers = {"Authorization": f"Bearer {globex_auth['token']}"}

        # ACME queries about their own data
        resp = self.client.post(
            f"{self.base_url}/query",
            json={"query": "Who is the CEO of ACME?"},
            headers=acme_headers,
        )
        acme_result = resp.json()

        # Globex queries about ACME data (should not find it)
        resp = self.client.post(
            f"{self.base_url}/query",
            json={"query": "Who is the CEO of ACME?"},
            headers=globex_headers,
        )
        globex_result = resp.json()

        if acme_result.get("success") and globex_result.get("success"):
            acme_chunks = acme_result["data"].get("retrieved_chunks", [])
            globex_chunks = globex_result["data"].get("retrieved_chunks", [])

            # ACME should find their data
            acme_has_data = any("ACME" in c.get("content", "") or "Jane" in c.get("content", "") for c in acme_chunks)
            self._assert("  ACME finds own data", acme_has_data)

            # Globex should NOT find ACME's data
            globex_has_acme = any("ACME" in c.get("content", "") or "Jane Smith" in c.get("content", "") for c in globex_chunks)
            self._assert("  Globex cannot see ACME data", not globex_has_acme)

        elif acme_result.get("success"):
            self._assert("  ACME query succeeded", True)
            self._assert("  Globex query failed (expected if no data)", not globex_result.get("success"))
        else:
            # Both failed — likely Qdrant retrieval issue
            self._assert(
                "  Queries failed (check Qdrant connection)",
                False,
            )
            if acme_result.get("error"):
                print(f"     ACME error: {acme_result['error'][:60]}")

    def _test_query_pipeline(self, auth: dict, tenant_name: str):
        """Test the full query pipeline."""
        print(f"🔍 Testing query pipeline for {tenant_name}...")

        headers = {"Authorization": f"Bearer {auth['token']}"}

        queries = [
            ("What does the company do?", "factual"),
            ("Who is the CEO?", "person"),
        ]

        for query_text, expected_intent in queries:
            resp = self.client.post(
                f"{self.base_url}/query",
                json={"query": query_text},
                headers=headers,
            )
            data = resp.json()

            if data.get("success"):
                result = data["data"]
                has_answer = result.get("answer") is not None
                has_intent = result.get("intent") is not None
                has_latency = result.get("latency_ms") is not None
                has_steps = result.get("step_latencies") is not None

                self._assert(
                    f"  \"{query_text[:30]}\" → intent={result.get('intent')}, "
                    f"chunks={len(result.get('retrieved_chunks', []))}, "
                    f"latency={result.get('latency_ms')}ms",
                    has_answer and has_intent and has_latency,
                )
            else:
                self._assert(
                    f"  \"{query_text[:30]}\" failed: {data.get('error', '')[:40]}",
                    False,
                )

    def _test_me_endpoint(self, auth: dict, tenant_name: str):
        """Test the /auth/me endpoint."""
        print(f"👤 Testing /auth/me...")

        headers = {"Authorization": f"Bearer {auth['token']}"}
        resp = self.client.get(f"{self.base_url}/auth/me", headers=headers)
        data = resp.json()

        if data.get("success"):
            me = data["data"]
            self._assert(f"  tenant_id matches", me["tenant_id"] == auth["tenant_id"])
            self._assert(f"  role is admin", me["role"] == "admin")
        else:
            self._assert(f"  /auth/me failed", False)

    # ----- Helpers -----

    def _assert(self, description: str, condition: bool):
        """Record a test assertion."""
        if condition:
            self.passed += 1
            print(f"  ✅ {description}")
        else:
            self.failed += 1
            self.errors.append(description)
            print(f"  ❌ {description}")

    def _print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  Total:  {total}")
        print(f"  Passed: {self.passed} ✅")
        print(f"  Failed: {self.failed} ❌")
        print()

        if self.failed > 0:
            print("  Failed tests:")
            for err in self.errors:
                print(f"    - {err}")
            print()
            print("❌ INTEGRATION TEST FAILED")
            sys.exit(1)
        else:
            print("🎉 ALL INTEGRATION TESTS PASSED!")


def main():
    parser = argparse.ArgumentParser(description="IntentRAG integration test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    test = IntegrationTest(args.url)
    test.run()


if __name__ == "__main__":
    main()
