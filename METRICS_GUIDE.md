# Enhanced Intent Classifier Metrics

## Overview

This evaluation framework provides comprehensive metrics for the intent classifier, including:

1. **Confusion Matrix** - Which intents are commonly confused
2. **Per-Intent F1 Scores** - Identify weak intent categories  
3. **Confidence Calibration** - Assess model confidence reliability
4. **Cross-Dataset Performance** - Compare TREC vs SQuAD vs SciQ

## Files

- **`evaluate_intent_classifier.py`** - Full evaluation script (requires trained model)
- **`demo_evaluation_metrics.py`** - Demo with synthetic data (runs without model)

## Quick Start

### 1. View Demo (No Model Required)

```bash
python3 demo_evaluation_metrics.py
```

This generates visualizations with synthetic data in `./demo_evaluation_results/`

### 2. Run Full Evaluation (Model Required)

First, train the model:
```bash
python3 train_intent_classifier_roberta.py --epochs 3
```

Then evaluate:
```bash
python3 evaluate_intent_classifier.py \
  --model_path ./intent_classifier_model_roberta \
  --output_dir ./evaluation_results
```

## Metrics Explained

### 1. Confusion Matrix

**Purpose**: Identify which intent classes are commonly confused

**Visualization**: Heatmap showing predicted vs true intents (row-normalized)

**Key Insights**:
- Diagonal = correct predictions
- Off-diagonal = confusion between intents
- Helps identify systematic misclassifications

**Example findings**:
```
High confusion between:
  • "What/Which (Factual)" ↔ "Other"
  • "Why/How (Explanation)" ↔ "What/Which (Factual)"
```

### 2. Per-Intent F1 Scores

**Purpose**: Identify weak intent categories that need improvement

**Metrics per intent**:
- **Precision**: How many predicted X were actually X?
- **Recall**: How many actual X were correctly identified?
- **F1-Score**: Harmonic mean of precision & recall
- **Support**: Number of samples in test set

**Example output**:
```
Per-Class F1 Scores:
  [0] What/Which (Factual)        F1: 0.7200  (n=800)
  [1] Who (Person)                F1: 0.6500  (n=150)  ⚠️ Weak
  [2] When (Time)                 F1: 0.5800  (n=200)  ⚠️ Very Weak
  [3] Where (Location)            F1: 0.6900  (n=180)
  [4] Why/How (Explanation)       F1: 0.7400  (n=650)
  [5] Other                       F1: 0.6200  (n=300)  ⚠️ Weak
```

**Action items**:
- Classes with F1 < 0.60: Need more training data or better features
- Classes with low support: May need data augmentation
- Compare across datasets to identify dataset-specific issues

### 3. Confidence Calibration

**Purpose**: Assess whether model confidence scores are reliable

**Metrics**:
- **Calibration Curve**: Perfect calibration = diagonal line
- **ECE (Expected Calibration Error)**: Average difference between confidence and accuracy
  - ECE < 0.05 = Well calibrated
  - ECE > 0.15 = Poorly calibrated
- **Confidence Distribution**:
  - Average confidence
  - Confidence on correct predictions
  - Confidence on incorrect predictions

**Ideal behavior**:
```
Confidence stats:
  Average:    0.75
  Correct:    0.82  ✓ Higher confidence on correct predictions
  Incorrect:  0.45  ✓ Lower confidence on errors
```

**Problematic patterns**:
- **Overconfident**: High confidence on incorrect predictions
- **Underconfident**: Low confidence even on correct predictions  
- **Poor separation**: Similar confidence for correct/incorrect

**Use cases**:
- Set confidence thresholds for routing
- Identify when to trigger fallback mechanisms
- Improve RAG pipeline reliability

### 4. Cross-Dataset Performance

**Purpose**: Compare performance across TREC, SQuAD, and SciQ

**Visualizations**:
- Overall metrics comparison (accuracy, precision, recall, F1)
- Per-class F1 across datasets
- Confidence distribution by dataset
- Sample distribution analysis

**Key questions answered**:
- Which dataset is hardest for the model?
- Are certain intents harder in specific datasets?
- Is the model biased toward certain datasets?
- Do we need to balance training data?

**Example insights**:
```
Dataset Performance:
  TREC:  Accuracy=54% (hardest - more diverse questions)
  SQuAD: Accuracy=68% (medium - context-based questions)
  SciQ:  Accuracy=72% (easiest - scientific domain focus)

Per-Intent Variance:
  "When (Time)": High variance across datasets → needs improvement
  "What/Which (Factual)": Consistent performance → robust
```

## Output Files

### Visualizations

All plots saved as high-resolution PNG (300 DPI):

- `confusion_matrix_trec.png`
- `confusion_matrix_squad.png`
- `confusion_matrix_sciq.png`
- `per_class_f1_comparison.png`
- `confidence_calibration.png`
- `cross_dataset_comparison.png`

### JSON Results

`evaluation_results.json` contains complete metrics:

```json
{
  "dataset": "TREC",
  "num_samples": 500,
  "accuracy": 0.54,
  "precision": 0.55,
  "recall": 0.54,
  "f1": 0.54,
  "confusion_matrix": [[...], ...],
  "per_class_metrics": {
    "What/Which (Factual)": {
      "precision": 0.72,
      "recall": 0.68,
      "f1": 0.70,
      "support": 150
    },
    ...
  },
  "confidence_stats": {
    "average": 0.75,
    "correct": 0.82,
    "incorrect": 0.45
  }
}
```

## Command-Line Options

### evaluate_intent_classifier.py

```bash
python3 evaluate_intent_classifier.py \
  --model_path ./intent_classifier_model_roberta \  # Model directory
  --output_dir ./evaluation_results \               # Output directory
  --max_samples 1000 \                              # Limit samples per dataset
  --no_plots                                        # Skip visualization generation
```

## Integration with RAG Pipeline

Use evaluation insights to optimize RAG:

### 1. Set Confidence Thresholds

Based on calibration analysis:

```python
# From confidence stats
CONFIDENCE_THRESHOLDS = {
    'high_confidence': 0.85,    # Use grounded RAG
    'medium_confidence': 0.60,  # Add verification step
    'low_confidence': 0.40      # Trigger fallback
}
```

### 2. Intent-Specific Routing

Based on per-class F1 scores:

```python
# Adjust retrieval strategy for weak intents
INTENT_STRATEGIES = {
    'When (Time)': {'top_k': 10, 'threshold': 0.6},  # Weak: retrieve more
    'What/Which (Factual)': {'top_k': 5, 'threshold': 0.75},  # Strong: standard
}
```

### 3. Dataset-Specific Handling

Based on cross-dataset performance:

```python
# TREC questions are harder - use more conservative approach
if query_complexity == 'high':  # TREC-like
    retrieval_params['top_k'] *= 2
    confidence_threshold += 0.1
```

## Improvement Strategies

### If Overall Accuracy < 60%

1. **More training epochs** (3 → 5+)
2. **Larger model** (roberta-base → roberta-large)
3. **Better data augmentation**
4. **Adjust class weights** for imbalanced classes

### If Specific Intent F1 < 50%

1. **Add more training examples** for that intent
2. **Review label mapping** - may be incorrect
3. **Check data quality** - mislabeled samples?
4. **Use intent-specific preprocessing**

### If Poorly Calibrated (ECE > 0.15)

1. **Temperature scaling** - post-training calibration
2. **Label smoothing** during training
3. **Confidence regularization** in loss function
4. **Ensemble methods** - average multiple predictions

### If Dataset Imbalance Issues

1. **Stratified sampling** for training
2. **Weighted loss** based on dataset
3. **Dataset-specific fine-tuning**
4. **Multi-task learning** approach

## Expected Benchmarks

Based on literature and similar tasks:

| Metric | Good | Excellent |
|--------|------|-----------|
| Overall Accuracy | 60-70% | 75%+ |
| Per-Class F1 (min) | 0.50+ | 0.65+ |
| Confidence ECE | < 0.10 | < 0.05 |
| Conf. Gap (correct-incorrect) | 0.15+ | 0.30+ |

## Troubleshooting

### Model won't load
```bash
# Ensure model path is correct
ls -la ./intent_classifier_model_roberta/
# Should contain: config.json, model.safetensors, tokenizer files
```

### Out of memory
```bash
# Reduce max_samples
python3 evaluate_intent_classifier.py --max_samples 500
```

### Dataset loading fails
```bash
# TREC requires datasets < 3.0.0
pip install 'datasets<3.0.0'
```

## Next Steps

1. ✅ **Run demo** to understand metrics
2. ✅ **Train model** (if not done)
3. ✅ **Run evaluation** and analyze results
4. ⏭️ **Implement improvements** based on findings
5. ⏭️ **Integrate with RAG pipeline** using insights
6. ⏭️ **Set up continuous monitoring** for production

## References

- Confusion Matrix: sklearn.metrics.confusion_matrix
- Calibration: [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- TREC Dataset: [Text REtrieval Conference](https://trec.nist.gov/)
- Evaluation Metrics: [Classification Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
