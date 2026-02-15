#!/usr/bin/env python3
"""
Setup Multi-Tenant Schema
-------------------------
Creates indexes on metadata fields for efficient filtering.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

collection_name = os.getenv('COLLECTION_NAME', 'knowledge_base')

print("="*70)
print("🔧 SETTING UP MULTI-TENANT SCHEMA")
print("="*70)
print()

# Create indexes for multi-tenant filtering
indexes_to_create = [
    ('company_id', PayloadSchemaType.KEYWORD, 'Company identifier'),
    ('department', PayloadSchemaType.KEYWORD, 'Department/team'),
    ('topic', PayloadSchemaType.KEYWORD, 'Topic/category'),
    ('access_level', PayloadSchemaType.KEYWORD, 'Access control level'),
    ('domain', PayloadSchemaType.KEYWORD, 'Domain (datascience, ml, etc)'),
]

print("Creating indexes for efficient filtering...")
print()

for field_name, field_type, description in indexes_to_create:
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_type
        )
        print(f"✅ Index created: {field_name} ({description})")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✓  Index exists: {field_name} ({description})")
        else:
            print(f"❌ Error creating {field_name}: {e}")

print()
print("="*70)
print("✅ MULTI-TENANT SCHEMA READY")
print("="*70)
print()
print("You can now:")
print("  1. Add company-specific documents with metadata")
print("  2. Query with CompanyContext for data isolation")
print("  3. Filter by company_id, department, topic, access_level")
print()
print("Example:")
print("  python3 multi_tenant_rag.py")
