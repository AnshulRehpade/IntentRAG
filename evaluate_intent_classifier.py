#!/usr/bin/env python3
"""
Enhanced Intent Classifier Evaluation
--------------------------------------
Comprehensive metrics analysis including:
1. Confusion Matrix: Which intents are commonly confused
2. Per-Intent F1 Scores: Identify weak intent categories
3. Confidence Calibration: Are confidence scores reliable?
4. Cross-Dataset Performance: TREC vs SQuAD vs SciQ breakdown
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)
from datasets import load_dataset

# Import preprocessing functions from training script
from train_intent_classifier_roberta import (
    load_trec_dataset,
    load_squad_dataset,
    load_sciq_dataset,
    preprocess_trec,
    preprocess_squad,
    preprocess_sciq
)


class IntentClassifierEvaluator:
    """Comprehensive evaluator for intent classifier."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            device: Device to run on (auto-detect if None)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        # Convert to absolute path to avoid HuggingFace validation issues
        model_path_abs = os.path.abspath(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_abs, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path_abs, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        label_mapping_path = os.path.join(model_path_abs, 'label_mapping.json')
        with open(label_mapping_path, 'r') as f:
            label_data = json.load(f)
            self.label_to_intent = {int(k): v for k, v in label_data['label_to_intent'].items()}
        
        self.intent_names = list(self.label_to_intent.values())
        print(f"✓ Model loaded on {self.device}")
        print(f"Intent classes: {self.intent_names}")
    
    def predict_with_confidence(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict intent labels with confidence scores.
        
        Args:
            texts: List of text queries
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, confidence_scores, all_logits)
        """
        predictions = []
        confidences = []
        all_logits = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()
                
                # Get predictions and confidence (max softmax probability)
                probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
                batch_preds = np.argmax(probs, axis=1)
                batch_conf = np.max(probs, axis=1)
                
                predictions.extend(batch_preds)
                confidences.extend(batch_conf)
                all_logits.append(logits)
        
        return (
            np.array(predictions),
            np.array(confidences),
            np.vstack(all_logits)
        )
    
    def evaluate_dataset(
        self,
        texts: List[str],
        labels: List[int],
        dataset_name: str
    ) -> Dict:
        """
        Evaluate on a single dataset.
        
        Args:
            texts: Input texts
            labels: True labels
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} ({len(texts)} samples)")
        print(f"{'='*60}")
        
        # Get predictions
        predictions, confidences, logits = self.predict_with_confidence(texts)
        
        # Convert to numpy
        labels = np.array(labels)
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        correct_mask = (predictions == labels)
        avg_conf_correct = np.mean(confidences[correct_mask]) if np.sum(correct_mask) > 0 else 0.0
        avg_conf_incorrect = np.mean(confidences[~correct_mask]) if np.sum(~correct_mask) > 0 else 0.0
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        print(f"\nConfidence Statistics:")
        print(f"  Average:    {avg_confidence:.4f}")
        print(f"  Correct:    {avg_conf_correct:.4f}")
        print(f"  Incorrect:  {avg_conf_incorrect:.4f}")
        
        print(f"\nPer-Class F1 Scores:")
        per_class_metrics = {}
        for i, intent_name in self.label_to_intent.items():
            if i < len(per_class_f1):
                f1_score = per_class_f1[i]
                support_count = per_class_support[i]
                print(f"  [{i}] {intent_name:25} F1: {f1_score:.4f}  (n={int(support_count)})")
                per_class_metrics[intent_name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(f1_score),
                    'support': int(support_count)
                }
        
        return {
            'dataset': dataset_name,
            'num_samples': len(texts),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'confidence_stats': {
                'average': float(avg_confidence),
                'correct': float(avg_conf_correct),
                'incorrect': float(avg_conf_incorrect)
            },
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'true_labels': labels.tolist()
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, dataset_name: str, save_path: str = None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            dataset_name: Name of dataset
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.intent_names,
            yticklabels=self.intent_names,
            cbar_kws={'label': 'Proportion'}
        )
        
        plt.title(f'Confusion Matrix - {dataset_name}\n(Row-normalized)', fontsize=14, fontweight='bold')
        plt.ylabel('True Intent', fontsize=12)
        plt.xlabel('Predicted Intent', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_per_class_f1(self, results: List[Dict], save_path: str = None):
        """
        Plot per-class F1 scores across datasets.
        
        Args:
            results: List of evaluation results
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        datasets = [r['dataset'] for r in results]
        x = np.arange(len(self.intent_names))
        width = 0.8 / len(datasets)
        
        # Plot bars for each dataset
        for i, result in enumerate(results):
            f1_scores = [
                result['per_class_metrics'].get(intent, {}).get('f1', 0.0)
                for intent in self.intent_names
            ]
            offset = (i - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, f1_scores, width, label=result['dataset'])
        
        ax.set_xlabel('Intent Class', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Per-Class F1 Scores Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.intent_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-class F1 plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confidence_calibration(self, results: List[Dict], save_path: str = None, num_bins: int = 10):
        """
        Plot confidence calibration curve.
        
        Args:
            results: List of evaluation results
            save_path: Path to save plot
            num_bins: Number of confidence bins
        """
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for ax, result in zip(axes, results):
            confidences = np.array(result['confidences'])
            predictions = np.array(result['predictions'])
            true_labels = np.array(result['true_labels'])
            
            # Create bins
            bins = np.linspace(0, 1, num_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Calculate accuracy per bin
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(num_bins):
                mask = (confidences >= bins[i]) & (confidences < bins[i+1])
                if i == num_bins - 1:  # Include 1.0 in last bin
                    mask = (confidences >= bins[i]) & (confidences <= bins[i+1])
                
                if np.sum(mask) > 0:
                    bin_acc = np.mean(predictions[mask] == true_labels[mask])
                    bin_conf = np.mean(confidences[mask])
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)
                    bin_counts.append(np.sum(mask))
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(bin_centers[i])
                    bin_counts.append(0)
            
            # Plot calibration curve
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
            ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model calibration', linewidth=2, markersize=8)
            
            # Add bar chart showing sample distribution
            ax2 = ax.twinx()
            ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.3, color='gray', label='Sample count')
            ax2.set_ylabel('Number of Samples', fontsize=10)
            ax2.tick_params(axis='y', labelsize=9)
            
            ax.set_xlabel('Confidence', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title(f'Calibration - {result["dataset"]}', fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            
            # Calculate Expected Calibration Error (ECE)
            ece = sum(abs(acc - conf) * count for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts))
            ece /= sum(bin_counts) if sum(bin_counts) > 0 else 1
            ax.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Calibration plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cross_dataset_comparison(self, results: List[Dict], save_path: str = None):
        """
        Plot cross-dataset performance comparison.
        
        Args:
            results: List of evaluation results
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        datasets = [r['dataset'] for r in results]
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        
        # 1. Overall metrics comparison
        ax = axes[0, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics))
        width = 0.8 / len(datasets)
        
        for i, (result, color) in enumerate(zip(results, colors)):
            values = [result[m] for m in metrics]
            offset = (i - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=result['dataset'], color=color)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Overall Metrics Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # 2. Confidence statistics
        ax = axes[0, 1]
        conf_metrics = ['average', 'correct', 'incorrect']
        x = np.arange(len(conf_metrics))
        
        for i, (result, color) in enumerate(zip(results, colors)):
            values = [result['confidence_stats'][m] for m in conf_metrics]
            offset = (i - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=result['dataset'], color=color)
        
        ax.set_ylabel('Confidence', fontsize=11)
        ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Average', 'Correct', 'Incorrect'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # 3. Per-class average F1
        ax = axes[1, 0]
        intent_f1_avg = defaultdict(list)
        
        for result in results:
            for intent, metrics in result['per_class_metrics'].items():
                intent_f1_avg[intent].append(metrics['f1'])
        
        intents = list(intent_f1_avg.keys())
        avg_f1s = [np.mean(intent_f1_avg[intent]) for intent in intents]
        std_f1s = [np.std(intent_f1_avg[intent]) for intent in intents]
        
        x = np.arange(len(intents))
        ax.bar(x, avg_f1s, yerr=std_f1s, capsize=5, alpha=0.7)
        ax.set_ylabel('Average F1 Score', fontsize=11)
        ax.set_title('Average Per-Class F1 (±std)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(intents, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # 4. Sample distribution
        ax = axes[1, 1]
        intent_counts = defaultdict(lambda: [0] * len(datasets))
        
        for i, result in enumerate(results):
            for intent, metrics in result['per_class_metrics'].items():
                intent_counts[intent][i] = metrics['support']
        
        intents = list(intent_counts.keys())
        x = np.arange(len(intents))
        
        for i, (dataset, color) in enumerate(zip(datasets, colors)):
            counts = [intent_counts[intent][i] for intent in intents]
            offset = (i - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, counts, width, label=dataset, color=color)
        
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title('Sample Distribution per Intent', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(intents, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cross-dataset comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def load_and_prepare_datasets(max_samples: int = None):
    """
    Load and prepare all three datasets for evaluation.
    
    Args:
        max_samples: Maximum samples per dataset (None for all)
        
    Returns:
        Dictionary with dataset splits
    """
    # Load datasets
    trec_data = load_trec_dataset()
    squad_data = load_squad_dataset()
    sciq_data = load_sciq_dataset()
    
    # Prepare TREC test set
    trec_test = trec_data['test'].map(preprocess_trec, remove_columns=trec_data['test'].column_names)
    if max_samples:
        trec_test = trec_test.select(range(min(max_samples, len(trec_test))))
    
    # Prepare SQuAD validation set
    squad_val = squad_data['validation'].map(preprocess_squad, remove_columns=squad_data['validation'].column_names)
    if max_samples:
        squad_val = squad_val.select(range(min(max_samples, len(squad_val))))
    
    # Prepare SciQ test set
    sciq_test = sciq_data['test'].map(preprocess_sciq, remove_columns=sciq_data['test'].column_names)
    if max_samples:
        sciq_test = sciq_test.select(range(min(max_samples, len(sciq_test))))
    
    return {
        'TREC': {'texts': trec_test['text'], 'labels': trec_test['label']},
        'SQuAD': {'texts': squad_val['text'], 'labels': squad_val['label']},
        'SciQ': {'texts': sciq_test['text'], 'labels': sciq_test['label']}
    }


def main():
    parser = argparse.ArgumentParser(description='Enhanced Intent Classifier Evaluation')
    parser.add_argument(
        '--model_path',
        type=str,
        default='./intent_classifier_model_roberta',
        help='Path to trained model (default: ./intent_classifier_model_roberta)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results (default: ./evaluation_results)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples per dataset (default: None, use all)'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("ENHANCED INTENT CLASSIFIER EVALUATION")
    print("="*80)
    
    # Initialize evaluator
    evaluator = IntentClassifierEvaluator(args.model_path)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_and_prepare_datasets(max_samples=args.max_samples)
    
    # Evaluate on each dataset
    all_results = []
    for dataset_name, data in datasets.items():
        result = evaluator.evaluate_dataset(
            texts=data['texts'],
            labels=data['labels'],
            dataset_name=dataset_name
        )
        all_results.append(result)
        
        # Plot confusion matrix for this dataset
        if not args.no_plots:
            cm = np.array(result['confusion_matrix'])
            cm_path = os.path.join(args.output_dir, f'confusion_matrix_{dataset_name.lower()}.png')
            evaluator.plot_confusion_matrix(cm, dataset_name, save_path=cm_path)
    
    # Generate cross-dataset visualizations
    if not args.no_plots:
        print("\nGenerating cross-dataset visualizations...")
        
        # Per-class F1 comparison
        f1_path = os.path.join(args.output_dir, 'per_class_f1_comparison.png')
        evaluator.plot_per_class_f1(all_results, save_path=f1_path)
        
        # Confidence calibration
        calib_path = os.path.join(args.output_dir, 'confidence_calibration.png')
        evaluator.plot_confidence_calibration(all_results, save_path=calib_path)
        
        # Cross-dataset comparison
        comparison_path = os.path.join(args.output_dir, 'cross_dataset_comparison.png')
        evaluator.plot_cross_dataset_comparison(all_results, save_path=comparison_path)
    
    # Save detailed results to JSON
    output_json = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Detailed results saved to {output_json}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    for result in all_results:
        print(f"\n{result['dataset']}:")
        print(f"  Samples:   {result['num_samples']}")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  Avg Conf:  {result['confidence_stats']['average']:.4f}")
        
        # Find weakest intent
        weakest = min(result['per_class_metrics'].items(), 
                     key=lambda x: x[1]['f1'])
        strongest = max(result['per_class_metrics'].items(),
                       key=lambda x: x[1]['f1'])
        print(f"  Weakest:   {weakest[0]} (F1: {weakest[1]['f1']:.4f})")
        print(f"  Strongest: {strongest[0]} (F1: {strongest[1]['f1']:.4f})")
    
    print("\n" + "="*80)
    print(f"✓ Evaluation complete! Results saved to {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
