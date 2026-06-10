#!/usr/bin/env python3.11
"""
Populate Qdrant knowledge base with sample ML/AI documents
"""
import os
from dotenv import load_dotenv
from knowledge_base import KnowledgeBaseBuilder

# Load environment variables
load_dotenv()

# Sample documents for the knowledge base
SAMPLE_DOCUMENTS = [
    {
        "text": "Pattern recognition is a branch of machine learning that focuses on identifying and classifying patterns in data. It involves extracting meaningful features from raw data and using them to make predictions or decisions. Pattern recognition is used in many applications including image recognition, speech recognition, and natural language processing.",
        "metadata": {"title": "Introduction to Pattern Recognition", "category": "ML Fundamentals", "source": "sample"}
    },
    {
        "text": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information. Deep learning uses neural networks with multiple layers to learn hierarchical representations. Applications include image classification, natural language processing, and game playing.",
        "metadata": {"title": "Neural Networks and Deep Learning", "category": "Deep Learning", "source": "sample"}
    },
    {
        "text": "Backpropagation is a fundamental algorithm for training neural networks. It works by computing gradients of the loss function with respect to network weights using the chain rule. These gradients are then used to update weights via gradient descent. Backpropagation enables efficient training of deep networks with multiple layers.",
        "metadata": {"title": "Backpropagation Algorithm", "category": "Training Methods", "source": "sample"}
    },
    {
        "text": "Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data, particularly images. They use convolutional layers to extract features and pooling layers to reduce dimensionality. CNNs have achieved state-of-the-art results in image classification, object detection, and semantic segmentation tasks.",
        "metadata": {"title": "Convolutional Neural Networks", "category": "Deep Learning", "source": "sample"}
    },
    {
        "text": "Recurrent Neural Networks (RNNs) are designed to process sequential data like time series or text. They maintain hidden states that capture information about previous inputs. Variants like LSTMs and GRUs address the vanishing gradient problem. RNNs are used for machine translation, speech recognition, and text generation.",
        "metadata": {"title": "Recurrent Neural Networks", "category": "Deep Learning", "source": "sample"}
    },
    {
        "text": "Transfer learning leverages pre-trained models on new tasks, reducing training time and data requirements. Fine-tuning involves training a pre-trained model on task-specific data with a low learning rate. This approach has enabled rapid progress in NLP with models like BERT and GPT, and in computer vision with ResNet and Vision Transformers.",
        "metadata": {"title": "Transfer Learning and Fine-tuning", "category": "Training Methods", "source": "sample"}
    },
    {
        "text": "Attention mechanisms allow models to focus on relevant parts of input data. In Transformers, self-attention enables parallel processing of sequences and captures long-range dependencies. Multi-head attention allows the model to attend to different representation subspaces. Attention has become fundamental to modern deep learning architectures.",
        "metadata": {"title": "Attention Mechanisms", "category": "Deep Learning", "source": "sample"}
    },
    {
        "text": "Transformers are a neural network architecture based entirely on attention mechanisms. They consist of encoder and decoder stacks with multi-head self-attention and feed-forward layers. Transformers enable parallel training and have superior scaling properties compared to RNNs. They form the basis of modern NLP models like BERT, GPT, and T5.",
        "metadata": {"title": "Transformer Architecture", "category": "Deep Learning", "source": "sample"}
    },
    {
        "text": "Word embeddings represent words as dense vectors in a continuous space. Methods like Word2Vec, GloVe, and FastText learn embeddings from large text corpora. Pre-trained embeddings capture semantic and syntactic relationships between words. Modern approaches use contextual embeddings from models like BERT that provide word representations dependent on context.",
        "metadata": {"title": "Word Embeddings and Representation Learning", "category": "NLP", "source": "sample"}
    },
    {
        "text": "NLP encompasses many tasks including sentiment analysis, machine translation, question answering, named entity recognition, and text summarization. Modern NLP systems use deep learning models, particularly Transformers. Transfer learning with pre-trained models like BERT and GPT has achieved significant improvements across NLP tasks.",
        "metadata": {"title": "Natural Language Processing Tasks", "category": "NLP", "source": "sample"}
    },
    {
        "text": "Machine learning enables systems to learn from data without explicit programming. Supervised learning learns from labeled examples, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through interaction with an environment. Key challenges include overfitting, underfitting, and data quality.",
        "metadata": {"title": "Machine Learning Fundamentals", "category": "ML Fundamentals", "source": "sample"}
    },
    {
        "text": "Gradient descent is the primary optimization algorithm for training machine learning models. Variants include batch gradient descent, stochastic gradient descent (SGD), and adaptive methods like Adam. Learning rate selection is crucial for convergence. Modern optimizers incorporate momentum, adaptive learning rates, and other techniques for improved performance.",
        "metadata": {"title": "Optimization and Gradient Descent", "category": "Training Methods", "source": "sample"}
    },
    {
        "text": "Regularization prevents overfitting by constraining model complexity. Techniques include L1/L2 regularization, dropout, batch normalization, and early stopping. Dropout randomly disables neurons during training to prevent co-adaptation. Batch normalization normalizes layer inputs to stabilize training and act as a regularizer.",
        "metadata": {"title": "Regularization Techniques", "category": "Training Methods", "source": "sample"}
    },
    {
        "text": "Evaluation metrics assess model performance on specific tasks. For classification: accuracy, precision, recall, F1-score, and AUC-ROC. For regression: MSE, RMSE, MAE, and R-squared. For NLP: BLEU, ROUGE, and task-specific metrics. Choosing appropriate metrics depends on the application and class imbalance.",
        "metadata": {"title": "Evaluation Metrics", "category": "Model Evaluation", "source": "sample"}
    },
    {
        "text": "Cross-validation estimates model performance by training on multiple data splits. K-fold cross-validation divides data into k folds for unbiased evaluation. Hyperparameter tuning optimizes learning rate, batch size, regularization strength, and architecture choices. Grid search and random search are common hyperparameter search strategies.",
        "metadata": {"title": "Cross-validation and Hyperparameter Tuning", "category": "Model Evaluation", "source": "sample"}
    }
]

def main():
    """Populate the knowledge base with sample documents"""
    
    # Get Qdrant credentials from environment
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
    
    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        print("Please configure your Qdrant Cloud credentials.")
        return
    
    # Initialize knowledge base builder
    kb = KnowledgeBaseBuilder(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name="knowledge_base"
    )
    
    print("=" * 70)
    print("POPULATING QDRANT KNOWLEDGE BASE WITH SAMPLE DOCUMENTS")
    print("=" * 70)
    print(f"\nProcessing {len(SAMPLE_DOCUMENTS)} sample documents...\n")
    
    # Chunk documents
    chunks = kb.chunk_documents(SAMPLE_DOCUMENTS)
    
    # Generate embeddings
    embeddings = kb.generate_embeddings(chunks)
    
    # Create collection (don't recreate if already exists)
    kb.create_collection(recreate=False)
    
    # Store in Qdrant
    stored_count = kb.store_in_qdrant(chunks, embeddings)
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully populated knowledge base with {stored_count} chunks")
    print("=" * 70)

if __name__ == "__main__":
    main()
