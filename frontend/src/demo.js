/**
 * Demo mode mock responses — used when backend is not available.
 * Simulates the full pipeline output for UI demonstration.
 */

const MOCK_QUERIES = {
  default: {
    intent: 'factual',
    intent_confidence: 0.92,
    answer: 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.',
    model: 'gpt-4o-mini',
    usage: { prompt_tokens: 380, completion_tokens: 95, total_tokens: 475 },
    retrieved_chunks: [
      { content: 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.', relevance_score: 0.94 },
      { content: 'There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.', relevance_score: 0.87 },
      { content: 'Deep learning is a subfield of machine learning that uses neural networks with many layers.', relevance_score: 0.72 },
    ],
    hallucination_check: {
      is_hallucinated: false,
      confidence_score: 0.89,
      retrieval_quality: 'good',
      entropy_score: 1.2,
      llm_verdict: 'SUPPORTED',
      hallucination_type: null,
      severity: null,
      flagged_claims: [],
      likely_causes: [],
      details: 'Confidence: 0.89 | Retrieval: good | LLM verdict: SUPPORTED | Entropy: 1.20 bits',
    },
    healing: { attempts: 1, was_healed: false, strategies_used: [], original_answer: null, improvement_reason: null },
    latency_ms: 1240,
  },
  person: {
    intent: 'person',
    intent_confidence: 0.97,
    answer: 'Python was created by Guido van Rossum. He began working on it in the late 1980s at Centrum Wiskunde & Informatica (CWI) in the Netherlands. He served as Python\'s Benevolent Dictator For Life (BDFL) until 2018.',
    model: 'gpt-4o-mini',
    usage: { prompt_tokens: 420, completion_tokens: 110, total_tokens: 530 },
    retrieved_chunks: [
      { content: 'Guido van Rossum is a Dutch programmer who created the Python programming language. He began working on Python in the late 1980s at Centrum Wiskunde & Informatica (CWI).', relevance_score: 0.96 },
      { content: 'He served as Python\'s Benevolent Dictator For Life (BDFL) until 2018 when he stepped down.', relevance_score: 0.91 },
    ],
    hallucination_check: {
      is_hallucinated: false,
      confidence_score: 0.93,
      retrieval_quality: 'good',
      entropy_score: 0.8,
      llm_verdict: 'SUPPORTED',
      hallucination_type: null,
      severity: null,
      flagged_claims: [],
      likely_causes: [],
      details: 'Confidence: 0.93 | Retrieval: good | LLM verdict: SUPPORTED | Entropy: 0.80 bits',
    },
    healing: { attempts: 1, was_healed: false, strategies_used: [], original_answer: null, improvement_reason: null },
    latency_ms: 980,
  },
  healed: {
    intent: 'time',
    intent_confidence: 0.88,
    answer: 'ChatGPT was released on November 30, 2022. It became the fastest-growing consumer application in history. [1]',
    model: 'gpt-4o-mini',
    usage: { prompt_tokens: 510, completion_tokens: 85, total_tokens: 595 },
    retrieved_chunks: [
      { content: '2022 - ChatGPT is released on November 30. Stable Diffusion is released in August.', relevance_score: 0.95 },
      { content: 'Sam Altman is the CEO of OpenAI. OpenAI released ChatGPT in November 2022, which became the fastest-growing consumer application in history.', relevance_score: 0.88 },
    ],
    hallucination_check: {
      is_hallucinated: false,
      confidence_score: 0.91,
      retrieval_quality: 'good',
      entropy_score: 1.1,
      llm_verdict: 'SUPPORTED',
      hallucination_type: null,
      severity: null,
      flagged_claims: [],
      likely_causes: [],
      details: 'Confidence: 0.91 | Retrieval: good | LLM verdict: SUPPORTED | Entropy: 1.10 bits',
    },
    healing: {
      attempts: 2,
      was_healed: true,
      strategies_used: ['strict_grounding'],
      original_answer: 'ChatGPT was released in late 2022, around October or November. It quickly gained over 100 million users in just two months.',
      improvement_reason: "Strategy 'strict_grounding' resolved the hallucination",
    },
    latency_ms: 2450,
  },
}

export function getMockQueryResponse(query) {
  const q = query.toLowerCase()

  // Match intent by keywords
  if (q.includes('who') || q.includes('person') || q.includes('created') || q.includes('founded')) {
    return { ...MOCK_QUERIES.person, query }
  }
  if (q.includes('when') || q.includes('date') || q.includes('released') || q.includes('year')) {
    return { ...MOCK_QUERIES.healed, query }
  }

  return { ...MOCK_QUERIES.default, query }
}

export const MOCK_HEALTH = {
  status: 'healthy',
  version: '2.0.0',
  services: { api: 'up', postgres: 'up', qdrant: 'up' },
}

export const MOCK_ANALYTICS = {
  summary: {
    total_queries: 147,
    total_hallucinations: 12,
    hallucination_rate: 0.082,
    clean_answers: 135,
    avg_latency_ms: 1380,
  },
  by_intent: [
    { intent: 'explanation', total_queries: 38, hallucinations: 7, hallucination_rate: 0.184, avg_latency_ms: 1820 },
    { intent: 'factual', total_queries: 52, hallucinations: 2, hallucination_rate: 0.038, avg_latency_ms: 1150 },
    { intent: 'person', total_queries: 28, hallucinations: 1, hallucination_rate: 0.036, avg_latency_ms: 1050 },
    { intent: 'time', total_queries: 15, hallucinations: 1, hallucination_rate: 0.067, avg_latency_ms: 1220 },
    { intent: 'location', total_queries: 8, hallucinations: 0, hallucination_rate: 0.0, avg_latency_ms: 980 },
    { intent: 'other', total_queries: 6, hallucinations: 1, hallucination_rate: 0.167, avg_latency_ms: 1650 },
  ],
  latency_comparison: {
    clean: { count: 135, avg_latency_ms: 1250, min_latency_ms: 420, max_latency_ms: 3200 },
    hallucinated: { count: 12, avg_latency_ms: 2680, min_latency_ms: 1800, max_latency_ms: 4100 },
  },
  recent_hallucinations: [
    { query: 'Explain how quantum computing breaks encryption', intent: 'explanation', answer_preview: 'Quantum computers use Shor\'s algorithm to factor large primes in polynomial time...', latency_ms: 3100, timestamp: '2025-06-08T14:22:00' },
    { query: 'What is the revenue of OpenAI?', intent: 'factual', answer_preview: 'OpenAI generated approximately $3.4 billion in revenue...', latency_ms: 2200, timestamp: '2025-06-07T09:15:00' },
  ],
  insights: [
    "'explanation' intent has the highest hallucination rate (18.4%). Consider adding more documents to data/explanation/.",
    "Hallucinated responses take 2.1x longer on average — self-healing retries add latency but improve accuracy.",
    "'location' intent has zero hallucinations — knowledge base coverage is strong here.",
  ],
}
