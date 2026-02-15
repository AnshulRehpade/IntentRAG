# Multi-Tenant RAG Architecture

## 🎯 Problem Statement

**Challenge**: How to handle company-specific data in a RAG system where:
- Multiple companies use the same system
- Each company has different policies, procedures, and domain knowledge
- Data must be isolated (Company A cannot see Company B's data)
- Topics may overlap (both companies have "A/B testing") but content differs
- LLM context must be company-specific to avoid cross-contamination

## 🏗️ Solution Architecture

### 1. Metadata-Based Data Isolation

**Strategy**: Use Qdrant's metadata filtering instead of separate collections

```python
# Every document gets company metadata
metadata = {
    'company_id': 'techcorp',           # REQUIRED - company identifier
    'department': 'data_science',       # Optional - department/team
    'topic': 'experimentation',         # Optional - topic/category
    'access_level': 'standard',         # Optional - access control
    'created_date': '2026-02-15',       # Optional - versioning
    'author': 'john@techcorp.com'      # Optional - tracking
}
```

**Benefits**:
- ✅ Single collection for all companies (easier management)
- ✅ Fast filtering using Qdrant's indexed metadata
- ✅ Flexible hierarchy (company → department → topic)
- ✅ Easy to add new companies without infrastructure changes

### 2. Context-Aware Querying

**Key Innovation**: `CompanyContext` dataclass defines query scope

```python
@dataclass
class CompanyContext:
    company_id: str                    # REQUIRED - which company
    department: Optional[str] = None   # Optional - narrow to department
    topic: Optional[str] = None        # Optional - narrow to topic
    access_level: str = "standard"     # Controls what user can see
```

**Query Flow**:
```
User Query → CompanyContext → Filter Builder → Qdrant Search (filtered) → LLM → Answer
                     ↓
              Only retrieves company-specific data
```

### 3. Multi-Level Filtering

**Hierarchy**:
```
Company (Required)
  ├── Department (Optional)
  │     ├── Topic (Optional)
  │     │     └── Access Level (Optional)
```

**Examples**:

```python
# Broadest: All company data
context = CompanyContext(company_id='techcorp')

# Department-specific: Only data science docs
context = CompanyContext(
    company_id='techcorp',
    department='data_science'
)

# Topic-specific: Only experimentation docs in data science
context = CompanyContext(
    company_id='techcorp',
    department='data_science',
    topic='experimentation'
)

# Access-controlled: Only public docs
context = CompanyContext(
    company_id='techcorp',
    access_level='restricted'  # Only sees public docs
)
```

## 🔒 Data Isolation Guarantees

### Critical Filter Implementation

```python
def _build_filter(self, context: CompanyContext) -> Filter:
    """Build Qdrant filter - ALWAYS includes company_id."""
    
    conditions = []
    
    # 🚨 CRITICAL: Always filter by company_id
    # This is the security boundary
    conditions.append(
        FieldCondition(
            key="company_id",
            match=MatchValue(value=context.company_id)
        )
    )
    
    # Add optional filters
    if context.department:
        conditions.append(
            FieldCondition(
                key="department",
                match=MatchValue(value=context.department)
            )
        )
    
    return Filter(must=conditions)  # ALL conditions must match
```

**Security Properties**:
1. **Impossible to query without company_id** - Required parameter
2. **Qdrant filters at vector search time** - No post-processing needed
3. **No data leakage** - Company A's docs never returned for Company B
4. **Auditable** - All queries logged with company context

## 📊 LLM Context Management

### Challenge
LLM has no inherent understanding of company boundaries. Without proper prompting, it might:
- Mix general knowledge with company-specific info
- Provide generic answers when company-specific ones exist
- Leak information across company boundaries

### Solution: Company-Aware Prompting

```python
prompt = f"""You are a company-specific AI assistant for {company_id}.

IMPORTANT: Only use information from the provided context. 
This context is specific to {company_id} and should not be 
mixed with general knowledge.

Context (from {company_id}'s knowledge base):
{context_text}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If not in context, say "Not available in {company_id}'s knowledge base"
3. Cite sources with [source: source_name, company: {company_id}]
4. Keep answers specific to {company_id}'s policies and procedures

Answer:"""
```

**Key Elements**:
1. **Explicit company identity** - LLM knows which company it's serving
2. **Context-only instruction** - No mixing with general knowledge
3. **Company-specific citations** - Sources include company ID
4. **Scoped disclaimers** - "Not in company KB" vs generic "I don't know"

## 🎨 Use Cases

### Use Case 1: Same Question, Different Companies

**Question**: "How do we conduct A/B testing?"

**TechCorp Context**:
```python
context = CompanyContext(company_id='techcorp')
```

**TechCorp Answer**:
> "At TechCorp, we run tests for 2 weeks with 10,000 users per variant. 
> We track conversion rate, revenue per user, and engagement metrics."

**HealthPlus Context**:
```python
context = CompanyContext(company_id='healthplus')
```

**HealthPlus Answer**:
> "HealthPlus conducts A/B testing following FDA guidelines. Tests run for 
> 3 months with 0.90 statistical power. Primary endpoint is patient outcomes."

**Result**: Same question, different answers based on company context! ✅

### Use Case 2: Department-Specific Knowledge

**Question**: "What's our data pipeline architecture?"

**Engineering Department**:
```python
context = CompanyContext(
    company_id='techcorp',
    department='data_engineering'
)
```
> "TechCorp uses Apache Airflow on AWS EMR with Spark. We use Great 
> Expectations for data quality validation."

**Data Science Department**:
```python
context = CompanyContext(
    company_id='techcorp',
    department='data_science'
)
```
> "This information is not available in data_science's knowledge base. 
> Try asking the data_engineering team."

### Use Case 3: Access Control

**Question**: "What's our customer churn prediction model?"

**Standard Access**:
```python
context = CompanyContext(
    company_id='techcorp',
    access_level='standard'
)
```
> "We use gradient boosting with customer behavior features. Model achieves 
> 85% AUC."

**Restricted Access** (external contractor):
```python
context = CompanyContext(
    company_id='techcorp',
    access_level='restricted'  # Only sees public docs
)
```
> "This information is not available in your access level."

## 🚀 Implementation Guide

### Step 1: Update Document Ingestion

**Add company metadata to all new documents**:

```python
from multi_tenant_rag import MultiTenantRAG

rag = MultiTenantRAG()

# Add company-specific document
rag.add_company_document(
    content="Company's AB testing procedure...",
    company_id="techcorp",
    metadata={
        'department': 'data_science',
        'topic': 'experimentation',
        'source': 'ab_testing_guide_v2',
        'access_level': 'standard',
        'created_date': '2026-02-15'
    }
)
```

### Step 2: Update Query Interface

**Replace generic queries with context-aware queries**:

```python
from multi_tenant_rag import MultiTenantRAG, CompanyContext

rag = MultiTenantRAG()

# Define user's context
context = CompanyContext(
    company_id='techcorp',
    department='data_science',  # Optional
    access_level='standard'     # Based on user role
)

# Query with context
result = rag.query(
    question="How do we run experiments?",
    context=context,
    verbose=True
)

print(result['answer'])
print(f"Sources: {result['sources']}")
```

### Step 3: Migrate Existing Data

**Add company metadata to existing documents**:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Get all points without company_id
all_points, _ = client.scroll(
    collection_name="knowledge_base",
    limit=1000,
    with_payload=True
)

# Update each point with company metadata
for point in all_points:
    # Determine company from source or manually assign
    company_id = "general"  # or extract from metadata
    
    # Update payload
    client.set_payload(
        collection_name="knowledge_base",
        payload={
            "company_id": company_id,
            "department": "general",
            "access_level": "standard"
        },
        points=[point.id]
    )
```

## 📈 Scaling Strategies

### Strategy 1: Single Collection (Recommended)

**Use when**:
- <100 companies
- <1M documents per company
- Need flexible cross-company queries (admin use case)

**Benefits**:
- Simple management
- Easy to add companies
- Flexible filtering

**Implementation**: Use metadata filtering (current approach)

### Strategy 2: Collection Per Company

**Use when**:
- >100 companies
- >1M documents per company
- Strict isolation required (regulatory)

**Benefits**:
- Physical separation
- Better performance at scale
- Easy to delete company data

**Implementation**:
```python
collection_name = f"kb_{company_id}"
client.search(collection_name=collection_name, ...)
```

### Strategy 3: Hybrid Approach

**Use when**:
- Mix of large and small companies
- Some companies need physical isolation

**Implementation**:
- Small companies: Shared collection with metadata filtering
- Large companies: Dedicated collections
- Routing logic determines which strategy per company

## 🔍 Monitoring & Analytics

### Track Usage Per Company

```python
# Log every query with context
query_log = {
    'timestamp': datetime.now(),
    'company_id': context.company_id,
    'department': context.department,
    'question': question,
    'retrieved_chunks': len(results),
    'top_score': results[0].score,
    'latency_ms': elapsed_time
}
```

### Company-Specific Metrics

```python
def get_company_health_metrics(company_id: str):
    return {
        'total_documents': count_documents(company_id),
        'avg_query_latency': calculate_avg_latency(company_id),
        'coverage_rate': questions_answered / total_questions,
        'avg_confidence': average_similarity_scores(company_id),
        'departments': list_departments(company_id),
        'top_topics': most_queried_topics(company_id)
    }
```

## 🎯 Best Practices

### 1. Always Require company_id
```python
# ❌ BAD: Optional company_id
def query(question: str, company_id: Optional[str] = None):
    ...

# ✅ GOOD: Required company_id
def query(question: str, context: CompanyContext):
    ...
```

### 2. Validate Company Access
```python
def query(question: str, context: CompanyContext, user_id: str):
    # Check if user belongs to company
    if not user_has_access(user_id, context.company_id):
        raise PermissionError(f"User {user_id} cannot access {context.company_id}")
    ...
```

### 3. Audit All Queries
```python
# Log every query for compliance
audit_log.info({
    'action': 'query',
    'user_id': user_id,
    'company_id': context.company_id,
    'question': question,
    'timestamp': datetime.now(),
    'ip_address': request.ip
})
```

### 4. Test Cross-Company Isolation
```python
def test_isolation():
    # Add doc to Company A
    rag.add_company_document(content="Secret A", company_id="company_a")
    
    # Query from Company B
    result = rag.query(
        "What is the secret?",
        CompanyContext(company_id="company_b")
    )
    
    # Should NOT find Company A's secret
    assert "Secret A" not in result['answer']
```

## 📚 Migration Checklist

- [ ] Update document ingestion to include company_id
- [ ] Add CompanyContext to all query endpoints
- [ ] Migrate existing documents with company metadata
- [ ] Update LLM prompts to be company-aware
- [ ] Add access control validation
- [ ] Implement audit logging
- [ ] Create company-specific dashboards
- [ ] Test cross-company isolation
- [ ] Document metadata schema
- [ ] Train team on multi-tenant usage

## 🎉 Summary

**The framework handles company-specific data by**:

1. **Metadata-based isolation** - Every document tagged with company_id
2. **Context-aware filtering** - Qdrant filters at query time
3. **Company-specific prompts** - LLM knows which company it's serving
4. **Hierarchical filtering** - Company → Department → Topic
5. **Access control** - User permissions enforced via context
6. **Audit logging** - Track all cross-company access

**Result**: Same RAG system serves multiple companies with complete data isolation! 🚀
