# Hallucination Detection & Analysis

## 🎯 Overview

The hallucination detection system automatically identifies when the RAG system generates information not supported by the retrieved context, analyzes root causes, and provides actionable insights for improvement.

---

## 🏗️ Architecture

```
RAG Pipeline
    ↓
Grounded Answer Generated
    ↓
Hallucination Detector (Multi-Method)
    ├─ Rule-based Checks
    │  ├─ Context length
    │  ├─ Citation presence
    │  └─ Explicit failure phrases
    └─ LLM Verification
       ├─ Claim extraction
       ├─ Evidence matching
       └─ Verdict: SUPPORTED/UNSUPPORTED
    ↓
HallucinationInstance
    ├─ Classification (Type, Severity)
    ├─ Root Cause Analysis
    └─ Logged to JSONL
    ↓
Periodic Analysis
    ├─ By Intent
    ├─ By Cause
    ├─ By Type/Severity
    └─ Context Correlation
```

---

## 📊 Hallucination Types

| Type | Description | Example |
|------|-------------|---------|
| **FACTUAL** | Facts not in context | "The model has 12B parameters" (context says 7B) |
| **ATTRIBUTION** | Wrong source attribution | Attributes info to wrong paper/source |
| **CONTRADICTION** | Contradicts context | Context: "released 2020", Answer: "released 2019" |
| **FABRICATION** | Invents details | Adds technical specs not mentioned |
| **EXTRAPOLATION** | Unsupported inference | Concludes trends not in data |

---

## ⚡ Severity Levels

- **MINOR**: Small details, doesn't affect main answer (e.g., wrong year by 1)
- **MODERATE**: Important details missing/wrong (e.g., key metric values)
- **SEVERE**: Core facts hallucinated (e.g., completely made-up information)

---

## 🔍 Root Cause Analysis

The system automatically identifies likely causes:

| Cause | Description | Fix |
|-------|-------------|-----|
| `INSUFFICIENT_CONTEXT` | Context < 200 chars | Add more documents to KB |
| `NO_CHUNKS_RETRIEVED` | Retrieval returned 0 results | Improve embeddings/indexing |
| `FEW_CHUNKS_RETRIEVED` | < 3 chunks retrieved | Lower score threshold |
| `LOW_RELEVANCE_SCORES` | Avg score < 0.3 | Improve query reformulation |
| `NO_CITATIONS` | Answer lacks source tags | Strengthen grounding prompt |
| `FALLBACK_MODE_USED` | General knowledge used | Expected behavior |
| `COMPLEX_EXPLANATION_QUERY` | "Why/How" questions harder | Add more explanation docs |
| `MODEL_EXTRAPOLATION` | Good context but still hallucinated | Stricter prompt engineering |

---

## 🚀 Usage

### 1. Automatic Detection (Integrated)

Hallucinations are automatically detected when using the RAG engine:

```python
from rag_engine import RAGEngine

# Initialize with hallucination detection enabled (default)
rag = RAGEngine(enable_hallucination_detection=True)

# Run query
response = rag.answer_query("How do transformers work?", verbose=True)

# Check results
if response.get("hallucination_detection", {}).get("has_hallucination"):
    print("⚠️ Hallucination detected!")
    print(f"Type: {response['hallucination_detection']['type']}")
    print(f"Causes: {response['hallucination_detection']['causes']}")
```

**Logs are automatically saved to:** `hallucination_log.jsonl`

### 2. Run Analysis

After collecting data, analyze patterns:

```bash
# Generate text report
python3 analyze_hallucinations.py

# Generate JSON report
python3 analyze_hallucinations.py --format json --output results.json

# Analyze specific log file
python3 analyze_hallucinations.py --log-file custom_log.jsonl
```

### 3. Streamlit Interface

The Streamlit app shows real-time hallucination detection:

```bash
streamlit run streamlit_app.py
```

- **Pills**: Show hallucination status (green/red badge)
- **Debug Mode**: View detailed detection results

---

## 📈 Analysis Reports

### Summary Statistics

```
Total Queries: 250
Hallucinations Detected: 18
Hallucination Rate: 7.2%
Clean Answers: 232
```

### By Intent

Shows which query types are most prone to hallucinations:

```
Intent          Hallucinations  Total  Rate
explanation     12/100          (12.0%)
factual         4/80            (5.0%)
person          2/40            (5.0%)
time            0/30            (0.0%)
```

**Insight**: Explanation queries have highest hallucination rate → Add more explanatory documents

### By Root Cause

```
Cause                          Count  Percentage
INSUFFICIENT_CONTEXT           8      44.4%
LOW_RELEVANCE_SCORES          5      27.8%
NO_CITATIONS                  3      16.7%
COMPLEX_EXPLANATION_QUERY     2      11.1%
```

**Insight**: Most hallucinations due to insufficient context → Expand knowledge base

### Context Quality Correlation

```
Hallucinated Answers:
  Avg Context Length: 187 chars
  Avg Chunks: 1.8
  Avg Score: 0.28
  Citation Rate: 33%

Clean Answers:
  Avg Context Length: 892 chars
  Avg Chunks: 4.2
  Avg Score: 0.61
  Citation Rate: 94%
```

**Insight**: Hallucinated answers have much shorter context and fewer chunks

---

## 🛠️ Integration Examples

### Manual Detection

Detect hallucinations for any query/answer pair:

```python
from hallucination_detector import HallucinationDetector

detector = HallucinationDetector()

result = detector.detect(
    query="What is backpropagation?",
    answer="Backpropagation is an algorithm...",
    context="Retrieved context here...",
    intent_name="explanation",
    intent_id=4,
    num_chunks=3,
    context_length=450,
    chunk_scores=[0.82, 0.75, 0.61]
)

if result.has_hallucination:
    print(f"Type: {result.hallucination_type}")
    print(f"Severity: {result.severity}")
    print(f"Causes: {result.likely_causes}")
```

### Batch Analysis

Analyze existing logs programmatically:

```python
from hallucination_detector import HallucinationAnalyzer

analyzer = HallucinationAnalyzer("hallucination_log.jsonl")

# Get summary
stats = analyzer.get_summary_stats()
print(f"Hallucination rate: {stats['hallucination_rate']:.1%}")

# Analyze by intent
by_intent = analyzer.analyze_by_intent()
for intent, data in by_intent.items():
    print(f"{intent}: {data['rate']:.1%}")

# Get worst cases
worst = analyzer.get_worst_cases(5)
for case in worst:
    print(f"Query: {case['query']}")
    print(f"Causes: {case['likely_causes']}")
```

---

## 🎯 Improvement Strategies

Based on analysis results, take action:

### 1. High Hallucination Rate (>10%)

**Actions:**
- Expand knowledge base with more documents
- Improve document chunking (smaller/larger chunks)
- Fine-tune embedding model on domain data
- Strengthen grounding prompt

### 2. Intent-Specific Issues

**Example**: Explanation queries have 15% hallucination rate

**Actions:**
- Add more explanatory documents (how-to guides, tutorials)
- Increase `top_k` for explanation intent (currently 8)
- Lower score threshold for explanation intent

### 3. Insufficient Context

**If 40%+ hallucinations caused by insufficient context:**

**Actions:**
- Add more documents to knowledge base
- Implement document upload feature
- Use hybrid search (semantic + keyword)
- Implement query expansion

### 4. Low Relevance Scores

**If avg score for hallucinated answers < 0.3:**

**Actions:**
- Improve query preprocessing
- Use better embedding model
- Implement re-ranking with cross-encoder
- Add query reformulation

### 5. Model Extrapolation

**If good context but still hallucinating:**

**Actions:**
- Make grounding prompt stricter
- Add explicit "DO NOT extrapolate" instructions
- Lower temperature (currently 0.3)
- Use different model (try GPT-4)

---

## 📊 Metrics Dashboard (Future)

Planned visualizations:

```python
# Hallucination rate over time
plot_hallucination_trend(analyzer)

# Heatmap: Intent vs Cause
plot_intent_cause_heatmap(analyzer)

# Context length distribution
plot_context_distribution(analyzer, split_by_hallucination=True)

# ROC curve for detection threshold tuning
plot_detection_roc(analyzer)
```

---

## 🔧 Configuration

### Disable Hallucination Detection

```python
# In code
rag = RAGEngine(enable_hallucination_detection=False)

# In .env
ENABLE_HALLUCINATION_DETECTION=false
```

### Custom Log File

```python
detector = HallucinationDetector()
detector.log_file = "custom_hallucination_log.jsonl"
```

### Detection Sensitivity

Adjust in `hallucination_detector.py`:

```python
# More aggressive detection (catch more potential issues)
detection_prompt += "\nBe VERY strict - mark as hallucination if ANY uncertainty."

# More lenient (only catch obvious hallucinations)
detection_prompt += "\nOnly flag CLEAR hallucinations - allow minor extrapolations."
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `hallucination_detector.py` | Core detection & analysis classes |
| `analyze_hallucinations.py` | CLI tool for generating reports |
| `hallucination_log.jsonl` | Auto-generated log file |
| `hallucination_report.txt` | Generated analysis report |
| `rag_engine.py` | Integration point (auto-detection) |
| `streamlit_app.py` | UI integration |

---

## 🎓 Best Practices

1. **Monitor Regularly**: Run analysis weekly to catch trends early
2. **Act on Insights**: Don't just collect data - use it to improve
3. **Iterate**: Test fixes, measure impact, repeat
4. **Context is King**: Most hallucinations stem from poor context
5. **Document Patterns**: Note recurring issues for systematic fixes

---

## 🔬 Example Analysis Session

```bash
# 1. Run queries to collect data
streamlit run streamlit_app.py
# (Use interface for 50+ queries)

# 2. Generate analysis
python3 analyze_hallucinations.py

# 3. Review report
cat hallucination_report.txt

# 4. Identify top cause (e.g., INSUFFICIENT_CONTEXT: 60%)

# 5. Take action
python3 populate_knowledge_base.py --add-documents ./new_docs/

# 6. Re-test
streamlit run streamlit_app.py
# (Test same queries)

# 7. Re-analyze
python3 analyze_hallucinations.py

# 8. Measure improvement
# Before: 12% hallucination rate
# After: 5% hallucination rate
# ✅ 58% reduction!
```

---

## 🚨 Known Limitations

1. **LLM-based detection**: Detector uses GPT-4o-mini, which can have false positives/negatives
2. **Cost**: Each detection call costs ~$0.0001 (500 tokens)
3. **Latency**: Adds ~1-2 seconds per query
4. **Fallback mode**: Detection disabled for fallback answers (they use general knowledge)

---

## 🔮 Future Enhancements

- [ ] Real-time dashboard with Grafana
- [ ] A/B testing framework for prompt changes
- [ ] Automated retraining triggers
- [ ] Slack/email alerts for high hallucination rates
- [ ] Fine-tuned hallucination classifier (faster, cheaper)
- [ ] Multi-language support
- [ ] Citation accuracy verification

---

## 📚 References

- [RAG Triad of Metrics](https://arxiv.org/abs/2309.15217) - Context relevance, groundedness, answer relevance
- [RAGAS Framework](https://github.com/explodinggradients/ragas) - RAG evaluation metrics
- [Hallucination Detection Survey](https://arxiv.org/abs/2309.05922)

---

**Questions?** Check logs in `hallucination_log.jsonl` or run `analyze_hallucinations.py --help`
