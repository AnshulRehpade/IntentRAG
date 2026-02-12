# IntentRAG 🎯

**Intent-Aware Retrieval-Augmented Generation System**

Production-ready RAG system combining intent classification with hybrid grounding for accurate, hallucination-resistant question answering.

## ✨ Features

- 🧠 **Intent Classification**: 99.93% accurate RoBERTa-based classifier
- 🔍 **Smart Retrieval**: Intent-based routing strategies
- 📚 **Vector Database**: Qdrant Cloud with semantic search
- 🎯 **Hybrid Grounding**: Context-based answers + intelligent fallback
- 🚨 **Hallucination Detection**: Real-time detection with root cause analysis
- 📊 **Comprehensive Metrics**: Full evaluation framework
- 🌐 **Web Interface**: Interactive Streamlit app

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt --break-system-packages
```

### 2. Configure Environment
Create `.env` file with:
```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key
COLLECTION_NAME=knowledge_base
OPENAI_API_KEY=sk-your_openai_key
```

### 3. Populate Knowledge Base
```bash
python3 populate_knowledge_base.py
```

### 4. Run Interface
```bash
streamlit run streamlit_app.py
```
Visit **http://localhost:8501**

## 💻 Usage

### Command Line
```bash
python3 rag_engine.py --query "How do transformers work?" --verbose
```

### Python API
```python
from rag_engine import RAGEngine

rag = RAGEngine()
response = rag.answer_query("How do transformers work?")
print(response['generation']['answer'])
```

## 🏗️ Architecture

```
Query → Intent Classification → Retrieval Router → Qdrant Search
           ↓                           ↓                  ↓
      (6 intents)              (Strategy selection)  (Top-k semantic)
                                       ↓
                              Context Processing
                                       ↓
                            Grounded Generation (GPT-4o-mini)
                                       ↓
                          Hallucination Detection
                                       ↓
                        Final Answer (Grounded/Fallback)
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Intent Accuracy** | 99.93% |
| **TREC** | 100% |
| **SQuAD** | 99.97% |
| **SciQ** | 99.90% |
| **Hallucination Rate** | <5% |
| **Latency** | ~2s end-to-end |

## 📁 Core Files

```
IntentRAG/
├── rag_engine.py                       # Main pipeline
├── rag_components.py                   # Supporting classes
├── knowledge_base.py                   # Qdrant operations
├── hallucination_detector.py           # Detection system
├── train_intent_classifier_roberta.py  # Training
├── evaluate_intent_classifier.py       # Evaluation
├── analyze_hallucinations.py           # Analysis
├── streamlit_app.py                    # Web UI
└── requirements.txt                    # Dependencies
```

## 🔧 Configuration

### Retrieval Strategies (rag_components.py)
```python
STRATEGIES = {
    "explanation": {"top_k": 8, "score_threshold": 0.25},
    "factual": {"top_k": 5, "score_threshold": 0.3},
}
```

### Intent Classes
- **0**: Factual (What/Which)
- **1**: Person (Who)
- **2**: Time (When)  
- **3**: Location (Where)
- **4**: Explanation (Why/How)
- **5**: Other

## 📊 Evaluation

### Intent Classifier
```bash
python3 evaluate_intent_classifier.py
```

### Hallucination Analysis
```bash
python3 analyze_hallucinations.py
```

## 📚 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: File organization
- **[CODEBASE_GUIDE.md](CODEBASE_GUIDE.md)**: Code walkthrough  
- **[METRICS_GUIDE.md](METRICS_GUIDE.md)**: Evaluation framework
- **[HALLUCINATION_DETECTION.md](HALLUCINATION_DETECTION.md)**: Detection system

## 🙏 Acknowledgments

- **Transformers**: Hugging Face
- **Vector DB**: Qdrant
- **LLM**: OpenAI GPT-4o-mini
- **Datasets**: TREC, SQuAD, SciQ

---

**Built with Intent Classification + RAG + Hallucination Detection**
