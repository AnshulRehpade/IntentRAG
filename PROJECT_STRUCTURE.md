# IntentRAG - Project Structure

## 📁 Core Files

### **Main Components**
- `rag_engine.py` - Main RAG orchestrator (intent → retrieval → generation)
- `rag_components.py` - Supporting classes (IntentClassifier, RAGRouter, ContextProcessor)
- `knowledge_base.py` - Vector database operations (Qdrant integration)
- `hallucination_detector.py` - Hallucination detection & logging

### **Training & Evaluation**
- `train_intent_classifier_roberta.py` - Train RoBERTa intent classifier
- `evaluate_intent_classifier.py` - Comprehensive evaluation with metrics
- `analyze_hallucinations.py` - Analyze hallucination patterns

### **Data & Utilities**
- `dataset_loaders.py` - Load datasets (TREC, SQuAD, SciQ, etc.)
- `populate_knowledge_base.py` - Add documents to Qdrant

### **Interface**
- `streamlit_app.py` - Interactive web interface

### **Configuration**
- `.env` - Environment variables (API keys, Qdrant URL)
- `requirements.txt` - Python dependencies
- `.gitignore` - Git exclusions

## 📚 Documentation
- `README.md` - Project overview and quick start
- `CODEBASE_GUIDE.md` - Detailed codebase walkthrough
- `METRICS_GUIDE.md` - Evaluation metrics framework
- `HALLUCINATION_DETECTION.md` - Hallucination detection system
- `CLEANUP_SUMMARY.md` - Project cleanup history
- `PROJECT_STRUCTURE.md` - This file

## 🗂️ Directories

### **Generated (Not in Git)**
- `intent_classifier_model_roberta/` - Trained model files
- `evaluation_results/` - Evaluation outputs
- `hallucination_log.jsonl` - Runtime hallucination logs
- `hallucination_report.txt` - Analysis reports
- `__pycache__/` - Python bytecode cache

## 🚀 Usage

### **1. Setup**
```bash
# Install dependencies
pip3 install -r requirements.txt --break-system-packages

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### **2. Train Model** (Optional - pre-trained model included)
```bash
python3 train_intent_classifier_roberta.py
```

### **3. Populate Knowledge Base**
```bash
python3 populate_knowledge_base.py
```

### **4. Run Interface**
```bash
streamlit run streamlit_app.py
```

### **5. Evaluate System**
```bash
# Evaluate intent classifier
python3 evaluate_intent_classifier.py

# Analyze hallucinations
python3 analyze_hallucinations.py
```

### **6. Use Programmatically**
```python
from rag_engine import RAGEngine

rag = RAGEngine()
response = rag.answer_query("How do transformers work?", verbose=True)
print(response['generation']['answer'])
```

## 📊 File Sizes (Approximate)

| File | Lines | Purpose |
|------|-------|---------|
| `rag_engine.py` | 514 | Main pipeline orchestration |
| `hallucination_detector.py` | 562 | Hallucination detection |
| `knowledge_base.py` | 483 | Vector DB operations |
| `train_intent_classifier_roberta.py` | 526 | Model training |
| `evaluate_intent_classifier.py` | 616 | Evaluation framework |
| `analyze_hallucinations.py` | 300+ | Analysis & reporting |
| `rag_components.py` | 182 | Supporting utilities |
| `streamlit_app.py` | 200+ | Web interface |

## 🔄 Typical Workflow

1. **Development**: Edit core files (`rag_engine.py`, `rag_components.py`)
2. **Testing**: Run queries through Streamlit interface
3. **Evaluation**: Check hallucination logs and metrics
4. **Improvement**: Adjust retrieval strategies, prompts, or thresholds
5. **Analysis**: Run `analyze_hallucinations.py` for insights

## 🧹 Clean Commands

```bash
# Remove cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Remove generated logs (keep for analysis)
# rm hallucination_log.jsonl hallucination_report.txt

# Remove evaluation results
# rm -rf evaluation_results/
```

## 📦 Dependencies

**Core**:
- `transformers` - Intent classification (RoBERTa)
- `qdrant-client` - Vector database
- `sentence-transformers` - Embeddings
- `openai` - Answer generation
- `streamlit` - Web interface

**Evaluation**:
- `scikit-learn` - Metrics
- `matplotlib`, `seaborn` - Visualizations
- `datasets` - Dataset loading

**Utilities**:
- `python-dotenv` - Environment variables
- `torch` - Deep learning backend
- `tiktoken` - Tokenization

## 🎯 Key Features

✅ Intent-based retrieval routing  
✅ Hybrid grounding (RAG + fallback)  
✅ Hallucination detection & logging  
✅ Comprehensive evaluation metrics  
✅ Interactive web interface  
✅ 99.93% intent classification accuracy  
✅ Production-ready error handling  

## 📝 Notes

- Model files (~500MB) excluded from git via `.gitignore`
- API keys stored in `.env` (not committed)
- Logs generated at runtime for analysis
- Evaluation results can be regenerated anytime
