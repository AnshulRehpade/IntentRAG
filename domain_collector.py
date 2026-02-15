#!/usr/bin/env python3
"""
Domain-Specific Data Collector
------------------------------
Automated data collection for building focused RAG systems.

Usage:
    python3 domain_collector.py --domain ml --target 300
    python3 domain_collector.py --domain software --target 500
"""

import os
import sys
import argparse
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import our document ingester
from add_documents import DocumentIngester


class DomainCollector:
    """Collects domain-specific documents automatically."""
    
    def __init__(self, domain: str, target_chunks: int = 200):
        """
        Initialize domain collector.
        
        Args:
            domain: Domain name (ml, software, business, healthcare, etc.)
            target_chunks: Target number of chunks to collect
        """
        self.domain = domain.lower()
        self.target_chunks = target_chunks
        self.ingester = DocumentIngester()
        self.collected_chunks = 0
        
        # Domain-specific sources
        self.sources = self._get_domain_sources()
    
    def _get_domain_sources(self) -> Dict:
        """Get curated sources for each domain."""
        
        sources = {
            'ml': {
                'name': 'Machine Learning / AI',
                'urls': [
                    'https://en.wikipedia.org/wiki/Deep_learning',
                    'https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)',
                    'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
                    'https://en.wikipedia.org/wiki/Convolutional_neural_network',
                    'https://en.wikipedia.org/wiki/Recurrent_neural_network',
                    'https://en.wikipedia.org/wiki/Long_short-term_memory',
                    'https://en.wikipedia.org/wiki/Natural_language_processing',
                    'https://en.wikipedia.org/wiki/Computer_vision',
                    'https://en.wikipedia.org/wiki/Reinforcement_learning',
                    'https://en.wikipedia.org/wiki/Generative_adversarial_network',
                ],
                'sample_files': ['ml']
            },
            
            'software': {
                'name': 'Software Engineering',
                'urls': [
                    'https://en.wikipedia.org/wiki/Software_design_pattern',
                    'https://en.wikipedia.org/wiki/SOLID',
                    'https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller',
                    'https://en.wikipedia.org/wiki/Representational_state_transfer',
                    'https://en.wikipedia.org/wiki/Microservices',
                    'https://en.wikipedia.org/wiki/Database_index',
                    'https://en.wikipedia.org/wiki/Version_control',
                    'https://en.wikipedia.org/wiki/Continuous_integration',
                    'https://en.wikipedia.org/wiki/Test-driven_development',
                    'https://en.wikipedia.org/wiki/Agile_software_development',
                ],
                'sample_files': ['software']
            },
            
            'business': {
                'name': 'Business & Management',
                'urls': [
                    'https://en.wikipedia.org/wiki/Business_model',
                    'https://en.wikipedia.org/wiki/Product_management',
                    'https://en.wikipedia.org/wiki/Customer_relationship_management',
                    'https://en.wikipedia.org/wiki/Key_performance_indicator',
                    'https://en.wikipedia.org/wiki/SWOT_analysis',
                    'https://en.wikipedia.org/wiki/Porter%27s_five_forces_analysis',
                    'https://en.wikipedia.org/wiki/Lean_startup',
                    'https://en.wikipedia.org/wiki/Objectives_and_key_results',
                    'https://en.wikipedia.org/wiki/Business_intelligence',
                    'https://en.wikipedia.org/wiki/Market_segmentation',
                ],
                'sample_files': ['business']
            },
            
            'finance': {
                'name': 'Finance & Investing',
                'urls': [
                    'https://en.wikipedia.org/wiki/Financial_statement',
                    'https://en.wikipedia.org/wiki/Discounted_cash_flow',
                    'https://en.wikipedia.org/wiki/Modern_portfolio_theory',
                    'https://en.wikipedia.org/wiki/Capital_asset_pricing_model',
                    'https://en.wikipedia.org/wiki/Options_(finance)',
                    'https://en.wikipedia.org/wiki/Technical_analysis',
                    'https://en.wikipedia.org/wiki/Fundamental_analysis',
                    'https://en.wikipedia.org/wiki/Cryptocurrency',
                    'https://en.wikipedia.org/wiki/Initial_public_offering',
                    'https://en.wikipedia.org/wiki/Venture_capital',
                ],
                'sample_files': []
            },
            
            'healthcare': {
                'name': 'Healthcare & Medicine',
                'urls': [
                    'https://en.wikipedia.org/wiki/Diabetes',
                    'https://en.wikipedia.org/wiki/Hypertension',
                    'https://en.wikipedia.org/wiki/Cardiovascular_disease',
                    'https://en.wikipedia.org/wiki/Cancer',
                    'https://en.wikipedia.org/wiki/Pharmacology',
                    'https://en.wikipedia.org/wiki/Evidence-based_medicine',
                    'https://en.wikipedia.org/wiki/Medical_diagnosis',
                    'https://en.wikipedia.org/wiki/Clinical_trial',
                    'https://en.wikipedia.org/wiki/Public_health',
                    'https://en.wikipedia.org/wiki/Epidemiology',
                ],
                'sample_files': []
            },
            
            'datascience': {
                'name': 'Data Science & Analytics',
                'urls': [
                    'https://en.wikipedia.org/wiki/Data_science',
                    'https://en.wikipedia.org/wiki/Data_analysis',
                    'https://en.wikipedia.org/wiki/Exploratory_data_analysis',
                    'https://en.wikipedia.org/wiki/Statistical_hypothesis_testing',
                    'https://en.wikipedia.org/wiki/Regression_analysis',
                    'https://en.wikipedia.org/wiki/Data_visualization',
                    'https://en.wikipedia.org/wiki/A/B_testing',
                    'https://en.wikipedia.org/wiki/Time_series',
                    'https://en.wikipedia.org/wiki/Feature_engineering',
                    'https://en.wikipedia.org/wiki/Data_preprocessing',
                    'https://en.wikipedia.org/wiki/Pandas_(software)',
                    'https://en.wikipedia.org/wiki/NumPy',
                    'https://en.wikipedia.org/wiki/Matplotlib',
                    'https://en.wikipedia.org/wiki/SQL',
                    'https://en.wikipedia.org/wiki/ETL',
                ],
                'sample_files': ['ml']  # Reuse ML samples as foundation
            }
        }
        
        return sources.get(self.domain, {
            'name': 'General',
            'urls': [],
            'sample_files': []
        })
    
    def collect(self) -> int:
        """
        Collect documents for the domain.
        
        Returns:
            Total chunks collected
        """
        print("="*70)
        print(f"🎯 DOMAIN-SPECIFIC DATA COLLECTION")
        print("="*70)
        print(f"Domain: {self.sources.get('name', self.domain)}")
        print(f"Target: {self.target_chunks} chunks")
        print("="*70)
        print()
        
        # Step 1: Add sample data if available
        if self.sources.get('sample_files'):
            print("📚 Step 1: Adding curated sample data...")
            for sample_type in self.sources['sample_files']:
                print(f"  Adding {sample_type} samples...")
                try:
                    chunks = self.ingester.add_sample_data(sample_type)
                    self.collected_chunks += chunks
                    print(f"  ✅ Added {chunks} chunks (Total: {self.collected_chunks})")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
            print()
        
        # Step 2: Scrape Wikipedia articles
        if self.sources.get('urls'):
            print("🌐 Step 2: Collecting from web sources...")
            print(f"  Found {len(self.sources['urls'])} URLs to process")
            print()
            
            # Check if web scraping dependencies are available
            try:
                import requests
                from bs4 import BeautifulSoup
                
                for i, url in enumerate(self.sources['urls'], 1):
                    if self.collected_chunks >= self.target_chunks:
                        print(f"\n✅ Target reached: {self.collected_chunks} chunks")
                        break
                    
                    print(f"  [{i}/{len(self.sources['urls'])}] {url.split('/')[-1][:50]}...")
                    try:
                        chunks = self.ingester.add_url(url)
                        self.collected_chunks += chunks
                        print(f"      ✅ Added {chunks} chunks (Total: {self.collected_chunks})")
                    except Exception as e:
                        print(f"      ❌ Failed: {str(e)[:50]}")
                
                print()
            
            except ImportError:
                print("  ⚠️  Web scraping not available")
                print("  Install: pip3 install requests beautifulsoup4 --break-system-packages")
                print()
        
        # Summary
        print("="*70)
        print("📊 COLLECTION SUMMARY")
        print("="*70)
        print(f"Total Chunks Collected: {self.collected_chunks}")
        print(f"Target Chunks: {self.target_chunks}")
        print(f"Progress: {(self.collected_chunks/self.target_chunks*100):.1f}%")
        
        if self.collected_chunks >= self.target_chunks:
            print("✅ Target achieved!")
        else:
            print(f"⚠️  Need {self.target_chunks - self.collected_chunks} more chunks")
            print()
            print("💡 Next steps:")
            print("  1. Add PDFs: python3 add_documents.py --file paper.pdf")
            print("  2. Add directory: python3 add_documents.py --directory ./docs --recursive")
            print("  3. Add more URLs: python3 add_documents.py --url https://...")
        
        print("="*70)
        
        return self.collected_chunks
    
    def get_recommendations(self):
        """Print recommendations for collecting more data."""
        recommendations = {
            'ml': [
                "📄 Download papers from ArXiv (https://arxiv.org/)",
                "📚 Add ML textbook chapters (Deep Learning by Goodfellow)",
                "🔗 Scrape tutorials from Towards Data Science",
                "📖 Add PyTorch/TensorFlow documentation",
                "🎓 Add course materials (Stanford CS229, fast.ai)",
            ],
            'software': [
                "📄 Add language documentation (Python, JavaScript)",
                "📚 Add design pattern books (Gang of Four)",
                "🔗 Scrape Stack Overflow top answers",
                "📖 Add framework guides (React, Django, Spring)",
                "🎓 Add coding best practices (Clean Code, Pragmatic Programmer)",
            ],
            'business': [
                "📄 Add business books (Lean Startup, Zero to One)",
                "📚 Add case studies (Harvard Business Review)",
                "🔗 Scrape PM blogs (Mind the Product, ProductPlan)",
                "📖 Add framework guides (OKRs, RICE, Jobs-to-be-Done)",
                "🎓 Add analytics guides (Amplitude, Mixpanel)",
            ],
            'finance': [
                "📄 Add finance textbooks (Corporate Finance, Securities Analysis)",
                "📚 Add investment guides (Graham, Buffett)",
                "🔗 Scrape financial news (Bloomberg, WSJ)",
                "📖 Add SEC filing examples (10-K, 10-Q)",
                "🎓 Add trading strategies and analysis methods",
            ],
            'healthcare': [
                "📄 Add medical textbooks (Harrison's, Gray's Anatomy)",
                "📚 Add clinical guidelines (WHO, CDC)",
                "🔗 Scrape health information (Mayo Clinic, WebMD)",
                "📖 Add drug databases (DrugBank)",
                "🎓 Add medical journal articles (PubMed)",
            ],
            'datascience': [
                "📄 Add data science books (Python for Data Analysis, Hands-On ML)",
                "📚 Add Kaggle competition notebooks and datasets documentation",
                "🔗 Scrape data science blogs (Towards Data Science, Analytics Vidhya)",
                "📖 Add pandas/numpy/matplotlib/scikit-learn documentation",
                "🎓 Add SQL tutorials and database optimization guides",
                "📊 Add statistics textbooks (Practical Statistics for Data Scientists)",
                "🔬 Add A/B testing guides (Optimizely, VWO)",
                "📈 Add data visualization best practices (Storytelling with Data)",
            ]
        }
        
        if self.domain in recommendations:
            print("\n💡 Recommendations for more data:")
            for rec in recommendations[self.domain]:
                print(f"  {rec}")


def main():
    parser = argparse.ArgumentParser(description='Collect domain-specific documents')
    
    parser.add_argument('--domain', type=str, required=True,
                       choices=['ml', 'software', 'business', 'finance', 'healthcare', 'datascience'],
                       help='Domain to collect data for')
    parser.add_argument('--target', type=int, default=200,
                       help='Target number of chunks to collect (default: 200)')
    parser.add_argument('--recommendations', action='store_true',
                       help='Show recommendations for collecting more data')
    
    args = parser.parse_args()
    
    collector = DomainCollector(args.domain, args.target)
    
    if args.recommendations:
        collector.get_recommendations()
    else:
        chunks_collected = collector.collect()
        
        if chunks_collected < args.target:
            print()
            collector.get_recommendations()


if __name__ == '__main__':
    main()
