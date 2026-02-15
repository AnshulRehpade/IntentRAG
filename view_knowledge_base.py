#!/usr/bin/env python3
"""
View Knowledge Base Inventory
Shows all chunks organized by source and domain
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import defaultdict

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

# Get all points from knowledge_base
result = client.scroll(
    collection_name='knowledge_base',
    limit=100,
    with_payload=True,
    with_vectors=False
)

points = result[0]

print('='*70)
print(f'📚 KNOWLEDGE BASE INVENTORY - {len(points)} CHUNKS')
print('='*70)
print()

# Group by source
by_source = defaultdict(list)

for point in points:
    source = point.payload.get('source', 'unknown')
    topic = point.payload.get('topic', 'N/A')
    domain = point.payload.get('domain', 'N/A')
    text_preview = point.payload.get('text', '')[:80].replace('\n', ' ')
    
    by_source[source].append({
        'id': point.id,
        'topic': topic,
        'domain': domain,
        'preview': text_preview
    })

# Print grouped by source
for source in sorted(by_source.keys()):
    chunks = by_source[source]
    domain = chunks[0]['domain']
    print(f'📁 {source} ({domain})')
    print(f'   Chunks: {len(chunks)}')
    for i, chunk in enumerate(chunks, 1):
        print(f'   [{i}] {chunk["preview"]}...')
    print()

# Summary by domain
print('='*70)
print('📊 SUMMARY BY DOMAIN')
print('='*70)
domain_counts = defaultdict(int)
for source, chunks in by_source.items():
    domain = chunks[0]['domain']
    domain_counts[domain] += len(chunks)

for domain, count in sorted(domain_counts.items()):
    print(f'  {domain}: {count} chunks')

print()
print(f'Total: {len(points)} chunks')
