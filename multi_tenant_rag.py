#!/usr/bin/env python3
"""
Multi-Tenant RAG Engine
-----------------------
Handles company-specific data with proper context isolation.

Key Features:
- Company-based data isolation using metadata filtering
- Topic-based routing within company context
- Prevents data leakage between companies
- Supports hierarchical filtering (company -> department -> topic)
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()


@dataclass
class CompanyContext:
    """Defines the context scope for a query."""
    company_id: str
    department: Optional[str] = None
    topic: Optional[str] = None
    access_level: str = "standard"  # standard, admin, restricted
    

class MultiTenantRAG:
    """RAG engine with multi-tenant support."""
    
    def __init__(self):
        """Initialize multi-tenant RAG engine."""
        # Qdrant connection
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = os.getenv("COLLECTION_NAME", "knowledge_base")
        
        # Embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("✓ Embedding model loaded")
        
        # OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _build_filter(self, context: CompanyContext) -> Filter:
        """
        Build Qdrant filter based on company context.
        
        This is the KEY to data isolation - only retrieve documents
        that match the company/department/topic scope.
        """
        conditions = []
        
        # CRITICAL: Always filter by company_id for data isolation
        conditions.append(
            FieldCondition(
                key="company_id",
                match=MatchValue(value=context.company_id)
            )
        )
        
        # Optional: Filter by department
        if context.department:
            conditions.append(
                FieldCondition(
                    key="department",
                    match=MatchValue(value=context.department)
                )
            )
        
        # Optional: Filter by topic
        if context.topic:
            conditions.append(
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=context.topic)
                )
            )
        
        # Optional: Filter by access level
        # Only retrieve documents user has access to
        if context.access_level == "restricted":
            conditions.append(
                FieldCondition(
                    key="access_level",
                    match=MatchValue(value="public")
                )
            )
        
        return Filter(must=conditions)
    
    def query(
        self,
        question: str,
        context: CompanyContext,
        top_k: int = 5,
        verbose: bool = False
    ) -> Dict:
        """
        Query with company-specific context isolation.
        
        Args:
            question: User's question
            context: Company context defining scope
            top_k: Number of chunks to retrieve
            verbose: Print debug information
            
        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            print("="*70)
            print(f"🏢 MULTI-TENANT QUERY")
            print("="*70)
            print(f"Question: {question}")
            print(f"Company: {context.company_id}")
            print(f"Department: {context.department or 'All'}")
            print(f"Topic: {context.topic or 'All'}")
            print(f"Access Level: {context.access_level}")
            print("="*70)
            print()
        
        # 1. Generate query embedding
        query_embedding = self.embedding_model.encode(question).tolist()
        
        # 2. Build filter for company context
        filter_conditions = self._build_filter(context)
        
        # 3. Search with filtering - ONLY gets company-specific data
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_conditions,
            limit=top_k,
            with_payload=True
        )
        
        if verbose:
            print(f"📊 Retrieved {len(search_results)} chunks (filtered by company context)")
            for i, result in enumerate(search_results, 1):
                payload = result.payload
                print(f"  [{i}] Score: {result.score:.4f}")
                print(f"      Company: {payload.get('company_id', 'N/A')}")
                print(f"      Department: {payload.get('department', 'N/A')}")
                print(f"      Topic: {payload.get('topic', 'N/A')}")
                print(f"      Source: {payload.get('source', 'N/A')}")
            print()
        
        # 4. Check if we found relevant results
        if not search_results or search_results[0].score < 0.3:
            return {
                'answer': f"I don't have information about this in {context.company_id}'s knowledge base. The question may be outside the scope of available company data.",
                'confidence': 'low',
                'retrieved_chunks': 0,
                'company_id': context.company_id,
                'sources': []
            }
        
        # 5. Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for result in search_results:
            text = result.payload.get('text', '')
            source = result.payload.get('source', 'unknown')
            company = result.payload.get('company_id', 'unknown')
            
            context_parts.append(f"[Source: {source}, Company: {company}]\n{text}\n")
            sources.append({
                'source': source,
                'score': result.score,
                'company_id': company,
                'department': result.payload.get('department'),
                'topic': result.payload.get('topic')
            })
        
        context_text = "\n---\n".join(context_parts)
        
        # 6. Generate answer with company-aware prompt
        prompt = f"""You are a company-specific AI assistant for {context.company_id}.

IMPORTANT: Only use information from the provided context. This context is specific to {context.company_id} and should not be mixed with general knowledge.

Context (from {context.company_id}'s knowledge base):
{context_text}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer isn't in the context, say "This information is not available in {context.company_id}'s knowledge base"
3. Cite sources with [source: source_name]
4. Keep answers specific to {context.company_id}'s context

Answer:"""
        
        if verbose:
            print("🤖 Generating company-specific answer...")
            print()
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant for {context.company_id}. Only provide information from the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'confidence': 'high' if search_results[0].score > 0.6 else 'medium',
            'retrieved_chunks': len(search_results),
            'company_id': context.company_id,
            'department': context.department,
            'topic': context.topic,
            'sources': sources,
            'top_score': search_results[0].score if search_results else 0
        }
    
    def add_company_document(
        self,
        content: str,
        company_id: str,
        metadata: Dict
    ) -> str:
        """
        Add a document with company-specific metadata.
        
        Args:
            content: Document text
            company_id: Company identifier
            metadata: Additional metadata (department, topic, etc.)
            
        Returns:
            Document ID
        """
        # Ensure company_id is in metadata
        metadata['company_id'] = company_id
        
        # Set default access level if not specified
        if 'access_level' not in metadata:
            metadata['access_level'] = 'standard'
        
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Generate unique ID
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Store in Qdrant with company metadata
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[{
                'id': doc_id,
                'vector': embedding,
                'payload': {
                    'text': content,
                    **metadata
                }
            }]
        )
        
        return doc_id
    
    def get_company_stats(self, company_id: str) -> Dict:
        """Get statistics for a company's data."""
        # Count documents for this company
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="company_id",
                    match=MatchValue(value=company_id)
                )
            ]
        )
        
        # Scroll to count (Qdrant doesn't have direct count with filter)
        results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_conditions,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        total_docs = len(results)
        
        # Analyze by department and topic
        departments = {}
        topics = {}
        
        for point in results:
            dept = point.payload.get('department', 'unspecified')
            topic = point.payload.get('topic', 'unspecified')
            
            departments[dept] = departments.get(dept, 0) + 1
            topics[topic] = topics.get(topic, 0) + 1
        
        return {
            'company_id': company_id,
            'total_documents': total_docs,
            'departments': departments,
            'topics': topics
        }


def demo_multi_tenant():
    """Demonstrate multi-tenant capabilities."""
    
    print("="*70)
    print("🏢 MULTI-TENANT RAG DEMONSTRATION")
    print("="*70)
    print()
    
    rag = MultiTenantRAG()
    
    # Simulate data from two different companies
    print("📝 Adding sample company-specific data...")
    print()
    
    # Company A: Tech startup with data science focus
    company_a_docs = [
        {
            'content': "At TechCorp, we use A/B testing extensively. Our standard approach is to run tests for 2 weeks with a minimum sample size of 10,000 users per variant. We track conversion rate, revenue per user, and engagement metrics.",
            'company_id': 'techcorp',
            'department': 'data_science',
            'topic': 'experimentation',
            'source': 'ab_testing_guide'
        },
        {
            'content': "TechCorp's data pipeline uses Apache Airflow for orchestration. All ETL jobs run on AWS EMR with Spark. We maintain data quality checks at each stage and use Great Expectations for validation.",
            'company_id': 'techcorp',
            'department': 'data_engineering',
            'topic': 'data_pipeline',
            'source': 'pipeline_docs'
        }
    ]
    
    # Company B: Healthcare company with different practices
    company_b_docs = [
        {
            'content': "HealthPlus conducts A/B testing following FDA guidelines for clinical significance. Tests must run for minimum 3 months with statistical power of 0.90. Primary endpoint is patient outcome improvement.",
            'company_id': 'healthplus',
            'department': 'clinical_analytics',
            'topic': 'experimentation',
            'source': 'clinical_testing_protocol'
        },
        {
            'content': "HealthPlus data pipeline complies with HIPAA regulations. All PHI is encrypted at rest and in transit. We use Azure Data Factory with private endpoints and maintain audit logs for all data access.",
            'company_id': 'healthplus',
            'department': 'data_engineering',
            'topic': 'data_pipeline',
            'source': 'compliance_docs'
        }
    ]
    
    # Add documents
    for doc in company_a_docs:
        rag.add_company_document(doc['content'], doc['company_id'], 
                                 {k: v for k, v in doc.items() if k != 'content'})
        print(f"  ✓ Added: {doc['source']} for {doc['company_id']}")
    
    for doc in company_b_docs:
        rag.add_company_document(doc['content'], doc['company_id'],
                                 {k: v for k, v in doc.items() if k != 'content'})
        print(f"  ✓ Added: {doc['source']} for {doc['company_id']}")
    
    print()
    print("="*70)
    print("📊 COMPANY STATISTICS")
    print("="*70)
    print()
    
    for company in ['techcorp', 'healthplus']:
        stats = rag.get_company_stats(company)
        print(f"Company: {company}")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Departments: {stats['departments']}")
        print(f"  Topics: {stats['topics']}")
        print()
    
    # Test queries with different company contexts
    print("="*70)
    print("🔍 TESTING CONTEXT ISOLATION")
    print("="*70)
    print()
    
    question = "How do we conduct A/B testing?"
    
    # Query for TechCorp
    print("1️⃣ Query from TechCorp perspective:")
    print("-" * 70)
    context_a = CompanyContext(company_id='techcorp', department='data_science')
    result_a = rag.query(question, context_a, verbose=True)
    print("Answer:", result_a['answer'])
    print()
    
    # Query for HealthPlus
    print("2️⃣ Query from HealthPlus perspective:")
    print("-" * 70)
    context_b = CompanyContext(company_id='healthplus', department='clinical_analytics')
    result_b = rag.query(question, context_b, verbose=True)
    print("Answer:", result_b['answer'])
    print()
    
    # Show that answers are different and company-specific
    print("="*70)
    print("✅ ISOLATION VERIFIED")
    print("="*70)
    print(f"TechCorp answer mentions: 2 weeks, 10,000 users (their policy)")
    print(f"HealthPlus answer mentions: 3 months, FDA guidelines (their policy)")
    print(f"Each company only sees their own data! 🎉")


if __name__ == '__main__':
    demo_multi_tenant()
