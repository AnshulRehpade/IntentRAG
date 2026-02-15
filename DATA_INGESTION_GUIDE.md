# Data Ingestion Guide

## 📊 Current Status

Knowledge Base: **23 chunks** (768D vectors, Cosine similarity)

## 🚀 Quick Commands

### Add Curated Sample Data
```bash
# ML/AI topics (Deep Learning, NLP, CV, RL, Model Evaluation)
python3 add_documents.py --sample ml

# Business topics (CRM, Analytics, BI)
python3 add_documents.py --sample business

# General knowledge (Communication, Soft Skills)
python3 add_documents.py --sample general
```

### Add Your Own Files

**Single Text/Markdown File:**
```bash
python3 add_documents.py --file path/to/document.txt
python3 add_documents.py --file path/to/notes.md
```

**Single PDF:**
```bash
# Requires: pip3 install PyPDF2 --break-system-packages
python3 add_documents.py --file path/to/paper.pdf
```

**From Web Page:**
```bash
# Requires: pip3 install requests beautifulsoup4 --break-system-packages
python3 add_documents.py --url "https://example.com/article"
```

**Entire Directory:**
```bash
# Add all .txt and .md files
python3 add_documents.py --directory ./documents

# Recursive (include subdirectories)
python3 add_documents.py --directory ./documents --recursive

# Custom extensions
python3 add_documents.py --directory ./documents --extensions .txt .md .rst
```

### Add Metadata
```bash
python3 add_documents.py --file document.txt --metadata '{"author": "John", "topic": "ML"}'
```

## 📚 Data Sources to Consider

### 1. **Research Papers (PDF)**
```bash
# Download papers from arXiv, Google Scholar, etc.
python3 add_documents.py --file transformers_paper.pdf
```

### 2. **Documentation**
```bash
# Technical docs, API references
python3 add_documents.py --directory ./docs --extensions .md .rst
```

### 3. **Knowledge Base Articles**
```bash
# Company wiki, Confluence exports, Notion pages
python3 add_documents.py --directory ./wiki --recursive
```

### 4. **Web Scraping**
```bash
# Blog posts, tutorials, Wikipedia articles
python3 add_documents.py --url "https://en.wikipedia.org/wiki/Machine_learning"
```

### 5. **Books (Text Format)**
```bash
# Textbooks, manuals (TXT format)
python3 add_documents.py --file ml_textbook.txt
```

## 🎯 Best Practices

### Content Quality
- ✅ **Use high-quality, factual content**
- ✅ **Diverse topics** for better coverage
- ✅ **Well-structured text** (headings, paragraphs)
- ❌ Avoid very short snippets (<100 words)
- ❌ Avoid duplicate content

### Optimal Size
- **Minimum**: 50-100 chunks for basic coverage
- **Good**: 200-500 chunks for solid performance
- **Excellent**: 1,000+ chunks for comprehensive system

### Chunking Strategy
- Default: 256 tokens/chunk with 25 token overlap
- Preserves context across chunks
- Automatic chunking by `add_documents.py`

## 📊 Monitor Your Knowledge Base

```bash
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

info = client.get_collection(os.getenv('COLLECTION_NAME'))
print(f'Total Chunks: {info.points_count}')
"
```

## 🔄 Training Data vs Knowledge Base

### Knowledge Base (Qdrant)
- **Purpose**: Provide context for RAG answers
- **Current**: 23 chunks
- **Add with**: `add_documents.py`
- **Impact**: Better retrieval, more accurate answers

### Intent Classifier Training Data
- **Purpose**: Train intent classification model
- **Current**: 83,784 samples (TREC, SQuAD, SciQ)
- **Training**: Already done (99.93% accuracy)
- **Retrain only if**: Adding new intent classes

## 💡 Example Workflows

### Academic Research Assistant
```bash
# Add ML papers
python3 add_documents.py --file bert_paper.pdf
python3 add_documents.py --file attention_is_all_you_need.pdf
python3 add_documents.py --sample ml
```

### Business Intelligence
```bash
# Add company docs
python3 add_documents.py --directory ./company_reports --recursive
python3 add_documents.py --sample business
```

### Technical Documentation
```bash
# Add API docs
python3 add_documents.py --directory ./api_docs --extensions .md .rst
python3 add_documents.py --url "https://docs.python.org/3/tutorial/"
```

## 🚀 Next Steps

1. **Add Domain-Specific Content**
   - Identify your use case (research, business, support, etc.)
   - Gather relevant documents
   - Use `add_documents.py` to ingest

2. **Test Retrieval Quality**
   ```bash
   streamlit run streamlit_app.py
   # Ask questions and check if relevant chunks are retrieved
   ```

3. **Iterate and Expand**
   - Monitor hallucination rate with `analyze_hallucinations.py`
   - Add more documents where gaps are found
   - Aim for 100+ chunks minimum

## 🛠️ Advanced: Custom Data Loader

Create custom loader for specific formats:

```python
from knowledge_base import KnowledgeBaseBuilder
import os
from dotenv import load_dotenv

load_dotenv()

kb = KnowledgeBaseBuilder(
    qdrant_url=os.getenv('QDRANT_URL'),
    qdrant_api_key=os.getenv('QDRANT_API_KEY'),
    collection_name=os.getenv('COLLECTION_NAME')
)

# Your custom documents
docs = [
    {'text': 'Your content here...', 'metadata': {'source': 'custom'}},
    # ... more documents
]

# Chunk
chunks = []
for doc in docs:
    doc_chunks = kb.chunker.chunk_text(doc['text'], doc['metadata'])
    chunks.extend(doc_chunks)

# Embed and store
embeddings = kb.generate_embeddings(chunks)
kb.store_in_qdrant(chunks, embeddings)
```

---

**Goal**: Grow from 23 chunks to 100+ chunks for production-ready performance! 🎯
