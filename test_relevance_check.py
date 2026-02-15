"""
Test script for relevance check functionality.
Tests with queries that have varying levels of context relevance.
"""

from rag_engine import RAGEngine

def test_relevance_check():
    """Test relevance check with different query types."""
    
    print("="*80)
    print("TESTING RELEVANCE CHECK FUNCTIONALITY")
    print("="*80)
    
    # Initialize RAG engine
    engine = RAGEngine()
    
    test_cases = [
        {
            "query": "How do I perform A/B testing?",
            "expected": "high_relevance",
            "description": "Query with highly relevant context in KB"
        },
        {
            "query": "What is the weather in San Francisco today?",
            "expected": "low_relevance",
            "description": "Query with no relevant context in KB"
        },
        {
            "query": "How do I use pandas for data manipulation?",
            "expected": "high_relevance",
            "description": "Query with relevant pandas documentation"
        },
        {
            "query": "Who won the Super Bowl in 2023?",
            "expected": "low_relevance",
            "description": "Sports query - not in data science KB"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test['description']}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print(f"{'='*80}")
        
        # Test WITH relevance check
        print("\n--- WITH RELEVANCE CHECK ---")
        result_with = engine.answer_query(test['query'], verbose=True, check_relevance=True)
        
        print("\n📊 RESULTS:")
        print(f"Answer: {result_with['generation']['answer'][:200]}...")
        print(f"Fallback Used: {result_with['fallback_used']}")
        
        if 'relevance_check' in result_with:
            rel = result_with['relevance_check']
            print(f"Relevance Check:")
            print(f"  - Relevant chunks: {rel['num_relevant']}/{rel['num_relevant'] + rel['num_irrelevant']}")
            print(f"  - Avg relevance: {rel['avg_relevance']:.3f}")
            print(f"  - Is relevant: {rel['is_relevant']}")
        
        # Compare with WITHOUT relevance check
        print("\n--- WITHOUT RELEVANCE CHECK (for comparison) ---")
        result_without = engine.answer_query(test['query'], verbose=False, check_relevance=False)
        print(f"Answer: {result_without['generation']['answer'][:200]}...")
        print(f"Fallback Used: {result_without['fallback_used']}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    test_relevance_check()
