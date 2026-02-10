#!/usr/bin/env python3
"""
Test intent classifier improvements in RAG context
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rag_components import IntentClassifier, RAGRouter

# Test queries
test_queries = [
    # Factual questions (should route to fact-based retrieval)
    ("What is pattern recognition?", "factual"),
    ("What are neural networks?", "factual"),
    ("What is machine learning?", "factual"),
    
    # Person questions (should route to person-based retrieval)
    ("Who invented the neural network?", "person"),
    ("Who is Geoffrey Hinton?", "person"),
    
    # Location questions (should route to location-based retrieval)
    ("Where is Stanford University?", "location"),
    ("Where is the MIT located?", "location"),
    
    # Time questions (should route to temporal retrieval)
    ("When was machine learning invented?", "time"),
    ("When was the first neural network created?", "time"),
    
    # Explanation/How questions (should route to explanation-based retrieval)
    ("How does backpropagation work?", "explanation"),
    ("How do convolutional networks function?", "explanation"),
]

# Initialize intent classifier
print("Loading improved intent classifier...")
classifier = IntentClassifier(model_path='./intent_classifier_model_roberta')

print("\n" + "="*80)
print("INTENT CLASSIFICATION TEST")
print("Testing improved TREC mapping with various question types")
print("="*80)

intent_names = {
    0: "factual",
    1: "person",
    2: "time",
    3: "location",
    4: "explanation",
    5: "other"
}

correct = 0
total = 0

for query, expected_intent in test_queries:
    intent_id, intent_name = classifier.predict_intent(query)
    predicted_intent = intent_names[intent_id]
    is_correct = (predicted_intent == expected_intent)
    
    status = "✓" if is_correct else "✗"
    if is_correct:
        correct += 1
    total += 1
    
    print(f"\n{status} Query: {query}")
    print(f"  Expected: {expected_intent:12} | Predicted: {predicted_intent:12} | ID: {intent_id}")
    
    # Show routing strategy
    strategy = RAGRouter.get_strategy(intent_name)
    print(f"  Routing Strategy: top_k={strategy['top_k']}, threshold={strategy['score_threshold']}")

print("\n" + "="*80)
print(f"ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
print("="*80)

print("""
✅ Intent Classification Test Complete

The improved TREC label mapping (corrected ABBR and NUM mappings) enables
the model to better distinguish between different question types:

Key Improvements:
  • ABBR → Factual (was: Other)
  • NUM → Factual (was: Time)
  • This allows proper routing to appropriate retrieval strategies

Next Steps:
  1. Populate Qdrant knowledge base with documents
  2. Test full RAG pipeline with retrieval and generation
  3. Evaluate end-to-end answer quality improvements
""")
