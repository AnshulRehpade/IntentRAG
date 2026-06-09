"""
Intent classification service — wraps the fine-tuned RoBERTa classifier from v1.

Model: RoBERTa-base fine-tuned on TREC + SQuAD + SciQ
Labels: factual(0), person(1), time(2), location(3), explanation(4), other(5)
Location: ./intent_classifier_model_roberta/
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings

# Map numeric labels to intent category names used across the system
LABEL_TO_INTENT = {
    0: "factual",
    1: "person",
    2: "time",
    3: "location",
    4: "explanation",
    5: "other",
}

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = (
    settings.intent_model_path
    if settings.intent_model_path
    else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "intent_classifier_model_roberta",
    )
)


class IntentClassifier:
    """
    Wraps the fine-tuned RoBERTa model from v1.
    Classifies queries into: factual, person, time, location, explanation, other.

    Loads the model lazily on first classification request.
    Runs on CPU by default (fast enough for single-query inference).
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._tokenizer = None
        self._device = None
        self._label_mapping: dict[int, str] = LABEL_TO_INTENT
        self._loaded = False

    async def load_model(self):
        """Load the fine-tuned RoBERTa model and tokenizer from disk."""
        if self._loaded:
            return

        model_path = Path(self._model_path)

        if not model_path.exists():
            print(
                f"⚠️  Intent classifier model not found at {model_path}. "
                "Using fallback heuristic classifier."
            )
            self._loaded = True
            return

        # Select device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        self._model.to(self._device)
        self._model.eval()

        # Load label mapping from saved file (if available)
        label_mapping_file = model_path / "label_mapping.json"
        if label_mapping_file.exists():
            with open(label_mapping_file) as f:
                data = json.load(f)
                # The file has "label_to_intent" with human-readable names
                # We map those back to our short category names
                raw_mapping = data.get("label_to_intent", {})
                # Keep our system mapping — the file uses verbose names like
                # "What/Which (Factual)" which we map to "factual" etc.
                # Our LABEL_TO_INTENT is already correct.

        self._loaded = True
        print(f"✅ Intent classifier loaded on {self._device}")

    async def classify(self, query: str) -> dict:
        """
        Classify a query's intent.

        Args:
            query: The user's question/query text

        Returns:
            {
                "intent": str,       # e.g. "factual"
                "confidence": float,  # 0.0 - 1.0
                "all_scores": dict    # {intent_name: score} for all categories
            }
        """
        if not self._loaded:
            await self.load_model()

        # If model couldn't load, use heuristic fallback
        if self._model is None:
            return self._heuristic_classify(query)

        # Tokenize
        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Get prediction
        predicted_label = torch.argmax(probs).item()
        confidence = probs[predicted_label].item()

        # Build all_scores dict
        all_scores = {}
        for label_idx in range(len(probs)):
            intent_name = self._label_mapping.get(label_idx, "other")
            all_scores[intent_name] = round(probs[label_idx].item(), 4)

        intent = self._label_mapping.get(predicted_label, "other")

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def _heuristic_classify(self, query: str) -> dict:
        """
        Fallback heuristic classifier when the RoBERTa model isn't available.
        Uses keyword matching (same logic as v1 SQuAD/SciQ preprocessing).
        """
        query_lower = query.lower().strip()

        # Simple keyword-based classification
        if any(w in query_lower for w in ["who ", "who's", "whom ", "whose "]):
            intent = "person"
        elif any(w in query_lower for w in ["when ", "when's", "what time", "what date", "what year"]):
            intent = "time"
        elif any(w in query_lower for w in ["where ", "where's", "what place", "what location", "what country"]):
            intent = "location"
        elif any(w in query_lower for w in ["why ", "why's", "how ", "how's", "explain", "describe"]):
            intent = "explanation"
        elif any(w in query_lower for w in ["what ", "what's", "which ", "define "]):
            intent = "factual"
        else:
            intent = "other"

        # Build scores (heuristic gives binary confidence)
        all_scores = {cat: 0.0 for cat in self._label_mapping.values()}
        all_scores[intent] = 0.85  # Heuristic confidence

        return {
            "intent": intent,
            "confidence": 0.85,
            "all_scores": all_scores,
        }


# Singleton instance
intent_classifier = IntentClassifier()
