#!/usr/bin/env python3
"""
Document Ingestion Tool
-----------------------
Add documents from various sources to the knowledge base.

Supports:
- Text files (.txt, .md)
- PDFs (.pdf)
- Web pages (URLs)
- JSON/CSV data
- Directory bulk import
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment
load_dotenv()

from knowledge_base import KnowledgeBaseBuilder

# Optional dependencies
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


class DocumentIngester:
    """Ingests documents from various sources into knowledge base."""
    
    def __init__(self):
        """Initialize knowledge base builder."""
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("COLLECTION_NAME", "knowledge_base")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        
        self.kb = KnowledgeBaseBuilder(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name
        )
    
    def _add_documents_to_kb(self, docs: List[Dict]) -> int:
        """
        Add documents to knowledge base using proper workflow.
        
        Args:
            docs: List of documents with 'text' and 'metadata'
            
        Returns:
            Number of chunks added
        """
        try:
            # Chunk documents
            chunker = self.kb.chunker
            all_chunks = []
            for doc in docs:
                chunks = chunker.chunk_text(doc['text'], doc.get('metadata', {}))
                all_chunks.extend(chunks)
            
            print(f"  Generated {len(all_chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.kb.generate_embeddings(all_chunks)
            
            # Store in Qdrant
            num_stored = self.kb.store_in_qdrant(all_chunks, embeddings)
            
            return num_stored
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return 0
    
    def add_text_file(self, filepath: str, metadata: Optional[Dict] = None) -> int:
        """
        Add text/markdown file to knowledge base.
        
        Args:
            filepath: Path to text file
            metadata: Additional metadata
            
        Returns:
            Number of chunks added
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            default_metadata = {
                'source': Path(filepath).name,
                'type': 'text',
                'filepath': filepath
            }
            if metadata:
                default_metadata.update(metadata)
            
            docs = [{'text': content, 'metadata': default_metadata}]
            return self._add_documents_to_kb(docs)
        
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            return 0
    
    def add_pdf(self, filepath: str, metadata: Optional[Dict] = None) -> int:
        """
        Add PDF file to knowledge base.
        
        Args:
            filepath: Path to PDF file
            metadata: Additional metadata
            
        Returns:
            Number of chunks added
        """
        if not PDF_AVAILABLE:
            print("❌ PyPDF2 not installed. Run: pip3 install PyPDF2 --break-system-packages")
            return 0
        
        try:
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Extract text from all pages
                texts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        page_metadata = {
                            'source': Path(filepath).name,
                            'type': 'pdf',
                            'page': page_num + 1,
                            'filepath': filepath
                        }
                        if metadata:
                            page_metadata.update(metadata)
                        
                        texts.append({
                            'text': text,
                            'metadata': page_metadata
                        })
                
                return self._add_documents_to_kb(texts)
        
        except Exception as e:
            print(f"❌ Error reading PDF {filepath}: {e}")
            return 0
    
    def add_url(self, url: str, metadata: Optional[Dict] = None) -> int:
        """
        Add web page content to knowledge base.
        
        Args:
            url: Web page URL
            metadata: Additional metadata
            
        Returns:
            Number of chunks added
        """
        if not WEB_AVAILABLE:
            print("❌ requests/beautifulsoup4 not installed.")
            print("Run: pip3 install requests beautifulsoup4 --break-system-packages")
            return 0
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            default_metadata = {
                'source': url,
                'type': 'web',
                'title': soup.title.string if soup.title else url
            }
            if metadata:
                default_metadata.update(metadata)
            
            docs = [{'text': text, 'metadata': default_metadata}]
            return self._add_documents_to_kb(docs)
        
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return 0
    
    def add_directory(
        self,
        directory: str,
        extensions: List[str] = ['.txt', '.md'],
        recursive: bool = True,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add all files from directory to knowledge base.
        
        Args:
            directory: Path to directory
            extensions: File extensions to include
            recursive: Search subdirectories
            metadata: Additional metadata for all files
            
        Returns:
            Total number of chunks added
        """
        total_chunks = 0
        path = Path(directory)
        
        if not path.exists():
            print(f"❌ Directory not found: {directory}")
            return 0
        
        # Find all matching files
        if recursive:
            files = []
            for ext in extensions:
                files.extend(path.rglob(f'*{ext}'))
        else:
            files = []
            for ext in extensions:
                files.extend(path.glob(f'*{ext}'))
        
        print(f"📁 Found {len(files)} files in {directory}")
        
        for filepath in files:
            print(f"  Processing: {filepath.name}...", end=' ')
            
            if filepath.suffix == '.pdf':
                chunks = self.add_pdf(str(filepath), metadata)
            else:
                chunks = self.add_text_file(str(filepath), metadata)
            
            if chunks > 0:
                print(f"✅ {chunks} chunks")
                total_chunks += chunks
            else:
                print("❌ Failed")
        
        return total_chunks
    
    def add_sample_data(self, domain: str = 'ml') -> int:
        """
        Add curated sample documents for specific domains.
        
        Args:
            domain: Domain ('ml', 'business', 'general')
            
        Returns:
            Number of chunks added
        """
        if domain == 'ml':
            docs = self._get_ml_samples()
        elif domain == 'business':
            docs = self._get_business_samples()
        else:
            docs = self._get_general_samples()
        
        return self._add_documents_to_kb(docs)
    
    def _get_ml_samples(self) -> List[Dict]:
        """Get ML/AI sample documents."""
        return [
            {
                'text': '''
Deep Learning Fundamentals:
Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. These networks can learn hierarchical representations of data through backpropagation. Key architectures include:

1. Convolutional Neural Networks (CNNs): Specialized for processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies.

2. Recurrent Neural Networks (RNNs): Designed for sequential data like time series or text. Variants include LSTM and GRU which solve the vanishing gradient problem.

3. Transformers: Use self-attention mechanisms to process sequential data in parallel. They form the basis of modern NLP models like BERT and GPT.

Training deep networks requires:
- Large datasets
- Powerful GPUs/TPUs
- Regularization techniques (dropout, batch normalization)
- Optimization algorithms (Adam, SGD with momentum)
- Learning rate schedules
                ''',
                'metadata': {'source': 'dl_fundamentals', 'type': 'textbook', 'topic': 'deep_learning'}
            },
            {
                'text': '''
Natural Language Processing (NLP):
NLP focuses on enabling computers to understand, interpret, and generate human language. Modern NLP relies heavily on:

1. Tokenization: Breaking text into tokens (words, subwords, characters)
2. Embeddings: Dense vector representations of words (Word2Vec, GloVe, BERT embeddings)
3. Language Models: Statistical models predicting word sequences
4. Transformers: Architecture using self-attention for context-aware representations

Key NLP tasks:
- Sentiment Analysis: Determining emotional tone
- Named Entity Recognition: Identifying entities (person, location, organization)
- Machine Translation: Converting text between languages
- Question Answering: Extracting answers from documents
- Text Summarization: Creating concise summaries

Transfer learning through pre-trained models has revolutionized NLP, allowing fine-tuning on specific tasks with limited data.
                ''',
                'metadata': {'source': 'nlp_guide', 'type': 'textbook', 'topic': 'nlp'}
            },
            {
                'text': '''
Machine Learning Model Evaluation:
Proper evaluation is critical for assessing model performance and avoiding overfitting. Key concepts:

1. Train/Validation/Test Split: Separating data to evaluate generalization
   - Training: 70-80% for learning
   - Validation: 10-15% for hyperparameter tuning
   - Test: 10-15% for final evaluation

2. Cross-Validation: K-fold technique for robust evaluation with limited data

3. Metrics:
   Classification:
   - Accuracy: (TP + TN) / Total
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - F1-Score: Harmonic mean of precision and recall
   - ROC-AUC: Area under receiver operating characteristic curve
   
   Regression:
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score

4. Confusion Matrix: Visualizes classification performance across classes

5. Learning Curves: Plot training/validation metrics over epochs to detect overfitting
                ''',
                'metadata': {'source': 'ml_evaluation', 'type': 'textbook', 'topic': 'evaluation'}
            },
            {
                'text': '''
Computer Vision Techniques:
Computer vision enables machines to interpret and understand visual information. Core techniques:

1. Image Classification: Assigning labels to entire images
   - CNNs are standard architecture
   - ResNet, VGG, EfficientNet are popular models
   - Transfer learning from ImageNet is common

2. Object Detection: Locating and classifying multiple objects
   - YOLO (You Only Look Once): Real-time detection
   - R-CNN family: Region-based methods
   - Outputs: Bounding boxes + class labels

3. Semantic Segmentation: Pixel-level classification
   - U-Net: Encoder-decoder architecture
   - DeepLab: Atrous convolution for multi-scale
   - Applications: Medical imaging, autonomous driving

4. Image Augmentation: Increasing training data diversity
   - Rotation, flipping, cropping
   - Color jittering, brightness adjustment
   - Improves model robustness

5. Feature Extraction: Learning meaningful representations
   - Pre-trained CNN backbones
   - SIFT, HOG (traditional methods)
                ''',
                'metadata': {'source': 'computer_vision', 'type': 'textbook', 'topic': 'vision'}
            },
            {
                'text': '''
Reinforcement Learning Basics:
RL trains agents to make sequential decisions by interacting with environments. Key components:

1. Agent: The learner/decision maker
2. Environment: The world the agent interacts with
3. State (s): Current situation
4. Action (a): Choices available to agent
5. Reward (r): Feedback signal
6. Policy (π): Strategy mapping states to actions

Core Algorithms:

Q-Learning:
- Value-based method
- Learns action-value function Q(s,a)
- Off-policy: learns optimal policy while following different policy
- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Policy Gradient:
- Direct policy optimization
- REINFORCE algorithm
- Well-suited for continuous action spaces

Actor-Critic:
- Combines value and policy methods
- Actor: Policy function
- Critic: Value function
- PPO, A3C are popular implementations

Applications:
- Game playing (AlphaGo, Dota 2)
- Robotics control
- Resource management
- Recommendation systems
                ''',
                'metadata': {'source': 'reinforcement_learning', 'type': 'textbook', 'topic': 'rl'}
            }
        ]
    
    def _get_business_samples(self) -> List[Dict]:
        """Get business domain sample documents."""
        return [
            {
                'text': '''
Customer Relationship Management (CRM):
CRM systems help businesses manage interactions with customers and prospects. Key components:

1. Contact Management: Centralized customer database
2. Sales Pipeline: Track opportunities through stages
3. Marketing Automation: Email campaigns, lead scoring
4. Customer Service: Ticket management, support history
5. Analytics: Insights into customer behavior and sales performance

Popular CRM platforms: Salesforce, HubSpot, Microsoft Dynamics, Zoho

Benefits:
- Improved customer retention
- Increased sales efficiency
- Better data-driven decision making
- Enhanced customer satisfaction
                ''',
                'metadata': {'source': 'crm_guide', 'type': 'business_doc', 'topic': 'crm'}
            },
            {
                'text': '''
Business Intelligence and Analytics:
BI transforms raw data into actionable insights for strategic decision-making.

Key Components:
1. Data Warehousing: Centralized storage for analysis
2. ETL (Extract, Transform, Load): Data preparation
3. Reporting: Standardized metrics and KPIs
4. Dashboards: Visual representations of key metrics
5. Predictive Analytics: Forecasting future trends

Tools: Tableau, Power BI, Looker, QlikView

Metrics to Track:
- Revenue and profitability
- Customer acquisition cost (CAC)
- Lifetime value (LTV)
- Churn rate
- Market share
- Operational efficiency
                ''',
                'metadata': {'source': 'bi_overview', 'type': 'business_doc', 'topic': 'analytics'}
            }
        ]
    
    def _get_general_samples(self) -> List[Dict]:
        """Get general knowledge sample documents."""
        return [
            {
                'text': '''
Effective Communication Skills:
Strong communication is essential in personal and professional settings.

Written Communication:
- Clear and concise language
- Proper grammar and punctuation
- Organized structure (intro, body, conclusion)
- Audience-appropriate tone

Verbal Communication:
- Active listening
- Clear articulation
- Appropriate pace and volume
- Non-verbal cues (body language, eye contact)

Presentation Skills:
- Know your audience
- Structure: Hook, content, call-to-action
- Visual aids (slides, charts)
- Practice and rehearsal
- Handle Q&A confidently
                ''',
                'metadata': {'source': 'communication', 'type': 'general_knowledge', 'topic': 'soft_skills'}
            }
        ]


def main():
    parser = argparse.ArgumentParser(description='Add documents to knowledge base')
    
    parser.add_argument('--file', type=str, help='Add single file (text, md, pdf)')
    parser.add_argument('--url', type=str, help='Add web page content')
    parser.add_argument('--directory', type=str, help='Add all files from directory')
    parser.add_argument('--extensions', type=str, nargs='+', 
                       default=['.txt', '.md'], help='File extensions to include')
    parser.add_argument('--recursive', action='store_true', 
                       help='Search subdirectories')
    parser.add_argument('--sample', type=str, choices=['ml', 'business', 'general'],
                       help='Add curated sample data')
    parser.add_argument('--metadata', type=str, help='Additional metadata (JSON format)')
    
    args = parser.parse_args()
    
    ingester = DocumentIngester()
    total_chunks = 0
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        import json
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in --metadata")
            return
    
    # Add single file
    if args.file:
        filepath = args.file
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        print(f"📄 Adding file: {filepath}")
        
        if filepath.endswith('.pdf'):
            chunks = ingester.add_pdf(filepath, metadata)
        else:
            chunks = ingester.add_text_file(filepath, metadata)
        
        total_chunks += chunks
    
    # Add from URL
    elif args.url:
        print(f"🌐 Fetching: {args.url}")
        chunks = ingester.add_url(args.url, metadata)
        total_chunks += chunks
    
    # Add from directory
    elif args.directory:
        print(f"📁 Scanning directory: {args.directory}")
        chunks = ingester.add_directory(
            args.directory,
            extensions=args.extensions,
            recursive=args.recursive,
            metadata=metadata
        )
        total_chunks += chunks
    
    # Add sample data
    elif args.sample:
        print(f"📚 Adding {args.sample} sample data...")
        chunks = ingester.add_sample_data(args.sample)
        total_chunks += chunks
    
    else:
        print("❌ No input specified. Use --help for options")
        return
    
    if total_chunks > 0:
        print(f"\n✅ Successfully added {total_chunks} chunks to knowledge base")
    else:
        print(f"\n❌ No chunks were added")


if __name__ == '__main__':
    main()
