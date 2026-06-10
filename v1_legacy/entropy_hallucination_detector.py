#!/usr/bin/env python3
"""
Enhanced Hallucination Detector with Entropy Analysis
------------------------------------------------------
Detects hallucinations using:
1. LLM verification (existing)
2. Token probability/entropy analysis (NEW)
3. Pattern-based detection

High entropy = Low confidence = Likely hallucination
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EntropyMetrics:
    """Token-level entropy and probability metrics."""
    avg_token_probability: float       # Average probability across all tokens
    min_token_probability: float       # Lowest probability token (red flag!)
    entropy: float                     # Information entropy (higher = more uncertain)
    high_entropy_tokens: List[Dict]    # Tokens with low probability
    suspicious_spans: List[Dict]       # Spans with multiple low-prob tokens
    softmax_bottlenecks: List[Dict]    # Sudden drops in confidence (NEW!)
    confidence_trajectory: List[float] # Probability over sequence (NEW!)
    

@dataclass
class HallucinationInstance:
    """Enhanced hallucination detection record."""
    timestamp: str
    query: str
    answer: str
    context: str
    intent_name: str
    intent_id: int
    
    # Hallucination classification
    has_hallucination: bool
    hallucination_type: Optional[str]
    severity: Optional[str]
    hallucinated_claims: List[str]
    confidence_score: float  # NEW: Overall confidence (0-1)
    
    # Entropy metrics (NEW)
    entropy_metrics: Optional[Dict]
    
    # Context quality metrics
    num_chunks_retrieved: int
    context_length: int
    avg_chunk_score: float
    min_chunk_score: float
    max_chunk_score: float
    
    # Additional metadata
    fallback_used: bool
    has_citations: bool
    answer_length: int
    
    # Root cause indicators
    likely_causes: List[str]
    
    def to_dict(self):
        return asdict(self)


class EnhancedHallucinationDetector:
    """Detects hallucinations with entropy analysis."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize detector with entropy support."""
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.log_file = "hallucination_log.jsonl"
        
        # Entropy thresholds
        self.HIGH_ENTROPY_THRESHOLD = 3.0      # Bits of entropy
        self.LOW_PROBABILITY_THRESHOLD = 0.1   # 10% probability
        self.SUSPICIOUS_TOKEN_THRESHOLD = 0.05 # 5% probability (very suspicious!)
    
    def analyze_entropy(
        self,
        answer: str,
        context: str
    ) -> Optional[EntropyMetrics]:
        """
        Analyze token probabilities and entropy in generated answer.
        
        Key insight: If model assigns low probability to tokens it generates,
        it's likely hallucinating rather than retrieving from context.
        
        Args:
            answer: Generated answer text
            context: Retrieved context
            
        Returns:
            EntropyMetrics or None if API doesn't support logprobs
        """
        try:
            # Re-generate answer with logprobs to analyze token confidence
            prompt = f"""Given this context, verify this answer.

Context:
{context}

Answer to verify:
{answer}

Respond with just 'VERIFIED' if answer is fully supported by context."""

            # Request with logprobs for entropy analysis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
                logprobs=True,  # Get token probabilities
                top_logprobs=5  # Get top 5 alternatives per token
            )
            
            if not response.choices[0].logprobs:
                return None
            
            # Extract token probabilities
            token_logprobs = []
            high_entropy_tokens = []
            probabilities_sequence = []  # Track confidence over sequence
            
            for token_info in response.choices[0].logprobs.content:
                logprob = token_info.logprob
                token = token_info.token
                probability = np.exp(logprob)
                
                token_logprobs.append(logprob)
                probabilities_sequence.append(probability)
                
                # Flag low-probability tokens (high entropy)
                if probability < self.LOW_PROBABILITY_THRESHOLD:
                    high_entropy_tokens.append({
                        'token': token,
                        'probability': probability,
                        'logprob': logprob,
                        'position': len(token_logprobs) - 1
                    })
            
            # Calculate entropy metrics
            probabilities = np.exp(token_logprobs)
            avg_prob = float(np.mean(probabilities))
            min_prob = float(np.min(probabilities))
            
            # Shannon entropy: H = -Σ(p * log2(p))
            entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
            
            # Find suspicious spans (consecutive low-prob tokens)
            suspicious_spans = self._find_suspicious_spans(high_entropy_tokens)
            
            # NEW: Detect softmax bottlenecks (sudden confidence drops)
            bottlenecks = self._detect_softmax_bottlenecks(
                probabilities_sequence,
                response.choices[0].logprobs.content
            )
            
            return EntropyMetrics(
                avg_token_probability=avg_prob,
                min_token_probability=min_prob,
                entropy=entropy,
                high_entropy_tokens=high_entropy_tokens,
                suspicious_spans=suspicious_spans,
                softmax_bottlenecks=bottlenecks,
                confidence_trajectory=probabilities_sequence
            )
            
        except Exception as e:
            print(f"Warning: Could not analyze entropy: {e}")
            return None
    
    def _detect_softmax_bottlenecks(
        self,
        probabilities: List[float],
        token_infos: List
    ) -> List[Dict]:
        """
        Detect sudden drops in confidence (softmax bottlenecks).
        
        Key insight: When model transitions from grounded content to hallucination,
        there's often a sharp drop in token probability. This marks the exact point
        where hallucination begins.
        
        Args:
            probabilities: Sequence of token probabilities
            token_infos: Token information from API
            
        Returns:
            List of detected bottlenecks with context
        """
        if len(probabilities) < 3:
            return []
        
        bottlenecks = []
        
        # Calculate probability deltas (changes between consecutive tokens)
        deltas = [probabilities[i] - probabilities[i-1] 
                 for i in range(1, len(probabilities))]
        
        # Look for significant drops
        for i, delta in enumerate(deltas):
            current_prob = probabilities[i+1]
            previous_prob = probabilities[i]
            
            # Criteria for bottleneck:
            # 1. Significant drop (>0.3 or >50% of previous probability)
            # 2. Drops into low-confidence region (<0.3)
            # 3. Previous tokens were confident (>0.6)
            
            is_significant_drop = (
                delta < -0.3 or  # Absolute drop
                (previous_prob > 0 and delta/previous_prob < -0.5)  # Relative drop
            )
            
            drops_to_low_confidence = current_prob < 0.3
            was_previously_confident = previous_prob > 0.6
            
            if is_significant_drop and drops_to_low_confidence and was_previously_confident:
                # Found a bottleneck!
                bottleneck = {
                    'position': i + 1,
                    'previous_token': token_infos[i].token,
                    'bottleneck_token': token_infos[i+1].token,
                    'previous_probability': previous_prob,
                    'bottleneck_probability': current_prob,
                    'confidence_drop': abs(delta),
                    'relative_drop_pct': abs(delta/previous_prob) * 100 if previous_prob > 0 else 0,
                    'context_before': ' '.join([token_infos[max(0, i-2)].token,
                                               token_infos[max(0, i-1)].token,
                                               token_infos[i].token]),
                    'context_after': ' '.join([token_infos[i+1].token] + 
                                             [token_infos[min(i+2, len(token_infos)-1)].token,
                                              token_infos[min(i+3, len(token_infos)-1)].token] 
                                             if i+2 < len(token_infos) else [])
                }
                bottlenecks.append(bottleneck)
        
        # Sort by confidence drop (most severe first)
        bottlenecks.sort(key=lambda x: x['confidence_drop'], reverse=True)
        
        return bottlenecks
    
    def _find_suspicious_spans(
        self,
        high_entropy_tokens: List[Dict]
    ) -> List[Dict]:
        """
        Find spans with multiple consecutive low-probability tokens.
        
        These are likely hallucinations (numbers, names, specific facts).
        """
        if not high_entropy_tokens:
            return []
        
        suspicious_spans = []
        current_span = []
        
        sorted_tokens = sorted(high_entropy_tokens, key=lambda x: x['position'])
        
        for i, token_info in enumerate(sorted_tokens):
            if not current_span:
                current_span.append(token_info)
            else:
                # Check if consecutive
                if token_info['position'] - current_span[-1]['position'] <= 2:
                    current_span.append(token_info)
                else:
                    # Save span if significant
                    if len(current_span) >= 2:
                        suspicious_spans.append({
                            'tokens': [t['token'] for t in current_span],
                            'avg_probability': np.mean([t['probability'] for t in current_span]),
                            'start_position': current_span[0]['position'],
                            'end_position': current_span[-1]['position']
                        })
                    current_span = [token_info]
        
        # Don't forget last span
        if len(current_span) >= 2:
            suspicious_spans.append({
                'tokens': [t['token'] for t in current_span],
                'avg_probability': np.mean([t['probability'] for t in current_span]),
                'start_position': current_span[0]['position'],
                'end_position': current_span[-1]['position']
            })
        
        return suspicious_spans
    
    def detect(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_chunks: List[Dict],
        intent_name: str = "unknown",
        intent_id: int = 0,
        fallback_used: bool = False
    ) -> HallucinationInstance:
        """
        Detect hallucinations using multi-method approach.
        
        Methods:
        1. LLM verification (is answer supported by context?)
        2. Entropy analysis (are token probabilities suspiciously low?)
        3. Pattern detection (citations, numbers, names)
        """
        
        # Method 1: LLM Verification
        verification_result = self._llm_verify(answer, context)
        
        # Method 2: Entropy Analysis (NEW)
        entropy_metrics = self.analyze_entropy(answer, context)
        
        # Method 3: Pattern-based detection
        has_citations = bool(re.search(r'\[source:', answer))
        has_numbers = bool(re.search(r'\b\d+\.?\d*\b', answer))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', answer))
        
        # Calculate context quality
        chunk_scores = [chunk.get('score', 0) for chunk in retrieved_chunks]
        avg_score = np.mean(chunk_scores) if chunk_scores else 0
        min_score = np.min(chunk_scores) if chunk_scores else 0
        max_score = np.max(chunk_scores) if chunk_scores else 0
        
        # Determine overall confidence
        confidence_score = self._calculate_confidence(
            verification_result,
            entropy_metrics,
            avg_score,
            has_citations
        )
        
        # Determine if hallucination occurred
        has_hallucination = (
            verification_result['support_level'] == 'UNSUPPORTED' or
            confidence_score < 0.5 or
            (entropy_metrics and entropy_metrics.entropy > self.HIGH_ENTROPY_THRESHOLD) or
            (entropy_metrics and len(entropy_metrics.softmax_bottlenecks) > 0)  # NEW: Bottleneck detected
        )
        
        # Identify likely causes
        likely_causes = self._identify_causes(
            verification_result,
            entropy_metrics,
            retrieved_chunks,
            avg_score,
            has_citations,
            fallback_used
        )
        
        # Determine severity
        severity = self._determine_severity(
            verification_result,
            confidence_score,
            entropy_metrics
        )
        
        instance = HallucinationInstance(
            timestamp=datetime.now().isoformat(),
            query=query,
            answer=answer,
            context=context,
            intent_name=intent_name,
            intent_id=intent_id,
            has_hallucination=has_hallucination,
            hallucination_type=verification_result.get('hallucination_type'),
            severity=severity,
            hallucinated_claims=verification_result.get('unsupported_claims', []),
            confidence_score=confidence_score,
            entropy_metrics=asdict(entropy_metrics) if entropy_metrics else None,
            num_chunks_retrieved=len(retrieved_chunks),
            context_length=len(context),
            avg_chunk_score=float(avg_score),
            min_chunk_score=float(min_score),
            max_chunk_score=float(max_score),
            fallback_used=fallback_used,
            has_citations=has_citations,
            answer_length=len(answer),
            likely_causes=likely_causes
        )
        
        # Log to JSONL
        self._log_instance(instance)
        
        return instance
    
    def _calculate_confidence(
        self,
        verification_result: Dict,
        entropy_metrics: Optional[EntropyMetrics],
        retrieval_score: float,
        has_citations: bool
    ) -> float:
        """
        Calculate overall confidence score (0-1).
        
        Combines:
        - LLM verification
        - Token probability/entropy
        - Retrieval quality
        - Citation presence
        """
        score = 0.0
        
        # LLM verification (40%)
        if verification_result['support_level'] == 'SUPPORTED':
            score += 0.4
        elif verification_result['support_level'] == 'PARTIALLY_SUPPORTED':
            score += 0.2
        
        # Entropy metrics (30%)
        if entropy_metrics:
            # Lower entropy = higher confidence
            if entropy_metrics.entropy < 2.0:
                score += 0.3
            elif entropy_metrics.entropy < 3.0:
                score += 0.15
            
            # High min probability = confident in all tokens
            if entropy_metrics.min_token_probability > 0.1:
                score += 0.1
        else:
            # No entropy data, give partial credit
            score += 0.15
        
        # Retrieval quality (20%)
        score += 0.2 * min(retrieval_score, 1.0)
        
        # Citations (10%)
        if has_citations:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_causes(
        self,
        verification_result: Dict,
        entropy_metrics: Optional[EntropyMetrics],
        retrieved_chunks: List[Dict],
        avg_score: float,
        has_citations: bool,
        fallback_used: bool
    ) -> List[str]:
        """Identify likely root causes of hallucination."""
        causes = []
        
        # Entropy-based causes
        if entropy_metrics:
            # Softmax bottleneck (NEW - highest priority indicator!)
            if entropy_metrics.softmax_bottlenecks:
                causes.append("SOFTMAX_BOTTLENECK_DETECTED")
                num_bottlenecks = len(entropy_metrics.softmax_bottlenecks)
                if num_bottlenecks > 1:
                    causes.append(f"MULTIPLE_CONFIDENCE_DROPS_{num_bottlenecks}")
            
            if entropy_metrics.entropy > self.HIGH_ENTROPY_THRESHOLD:
                causes.append("HIGH_ENTROPY_GENERATION")
            
            if entropy_metrics.min_token_probability < self.SUSPICIOUS_TOKEN_THRESHOLD:
                causes.append("VERY_LOW_TOKEN_CONFIDENCE")
            
            if len(entropy_metrics.high_entropy_tokens) > 5:
                causes.append("MULTIPLE_LOW_CONFIDENCE_TOKENS")
            
            if entropy_metrics.suspicious_spans:
                causes.append("SUSPICIOUS_TOKEN_SPANS")
        
        # Existing causes
        if avg_score < 0.5:
            causes.append("LOW_RELEVANCE_SCORES")
        
        if len(retrieved_chunks) < 3:
            causes.append("INSUFFICIENT_CONTEXT")
        
        if not has_citations:
            causes.append("MISSING_CITATIONS")
        
        if fallback_used:
            causes.append("FALLBACK_ANSWER_USED")
        
        if verification_result.get('contradictions'):
            causes.append("CONTRADICTORY_SOURCES")
        
        return causes
    
    def _determine_severity(
        self,
        verification_result: Dict,
        confidence_score: float,
        entropy_metrics: Optional[EntropyMetrics]
    ) -> str:
        """Determine hallucination severity."""
        
        if not verification_result.get('support_level') == 'UNSUPPORTED':
            return "MINOR"
        
        # Check entropy for severity
        if entropy_metrics:
            if entropy_metrics.min_token_probability < 0.01:
                return "SEVERE"  # Model is basically guessing
            elif entropy_metrics.entropy > 4.0:
                return "SEVERE"
        
        if confidence_score < 0.3:
            return "SEVERE"
        elif confidence_score < 0.5:
            return "MODERATE"
        else:
            return "MINOR"
    
    def _llm_verify(self, answer: str, context: str) -> Dict:
        """LLM-based verification (existing logic)."""
        
        prompt = f"""Analyze if this answer is supported by the context.

Context:
{context}

Answer:
{answer}

Classify the answer as:
- SUPPORTED: All claims are in the context
- PARTIALLY_SUPPORTED: Some claims are in context, some aren't
- UNSUPPORTED: Claims contradict or are not in context

Return JSON:
{{
    "support_level": "SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED",
    "hallucination_type": "FACTUAL|ATTRIBUTION|CONTRADICTION|FABRICATION|EXTRAPOLATION|null",
    "unsupported_claims": ["claim1", "claim2"],
    "contradictions": ["contradiction1"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            print(f"Warning: LLM verification failed: {e}")
            return {
                "support_level": "UNKNOWN",
                "hallucination_type": None,
                "unsupported_claims": [],
                "contradictions": []
            }
    
    def _log_instance(self, instance: HallucinationInstance):
        """Log hallucination instance to JSONL."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(instance.to_dict()) + '\n')
        except Exception as e:
            print(f"Warning: Could not log hallucination: {e}")


if __name__ == '__main__':
    # Demo entropy-based detection
    print("="*70)
    print("🔬 ENTROPY-BASED HALLUCINATION DETECTION DEMO")
    print("="*70)
    print()
    
    detector = EnhancedHallucinationDetector()
    
    # Test case: Answer with hallucinated specific numbers
    context = "Machine learning models require training data. The amount varies by problem."
    
    answer_hallucinated = "Machine learning models typically need 10,247 samples for good performance."
    
    print("Test Case: Hallucinated Specific Number")
    print("-" * 70)
    print(f"Context: {context}")
    print(f"Answer: {answer_hallucinated}")
    print()
    
    result = detector.detect(
        query="How much data do ML models need?",
        answer=answer_hallucinated,
        context=context,
        retrieved_chunks=[{'score': 0.7}],
        intent_name="factual"
    )
    
    print(f"Has Hallucination: {result.has_hallucination}")
    print(f"Confidence Score: {result.confidence_score:.3f}")
    print(f"Severity: {result.severity}")
    print(f"Likely Causes: {', '.join(result.likely_causes)}")
    
    if result.entropy_metrics:
        metrics = result.entropy_metrics
        print(f"\n📊 Entropy Metrics:")
        print(f"  Avg Token Probability: {metrics['avg_token_probability']:.3f}")
        print(f"  Min Token Probability: {metrics['min_token_probability']:.3f}")
        print(f"  Entropy: {metrics['entropy']:.3f} bits")
        print(f"  High Entropy Tokens: {len(metrics['high_entropy_tokens'])}")
        
        # NEW: Show softmax bottlenecks
        if metrics['softmax_bottlenecks']:
            print(f"\n🚨 SOFTMAX BOTTLENECKS DETECTED: {len(metrics['softmax_bottlenecks'])}")
            print("  (Sudden confidence drops marking hallucination start)")
            print()
            for i, bottleneck in enumerate(metrics['softmax_bottlenecks'][:3], 1):
                print(f"  Bottleneck #{i}:")
                print(f"    Position: {bottleneck['position']}")
                print(f"    Before: '{bottleneck['context_before']}'")
                print(f"    Drop at: '{bottleneck['bottleneck_token']}'")
                print(f"    After: '{bottleneck['context_after']}'")
                print(f"    Confidence: {bottleneck['previous_probability']:.3f} → {bottleneck['bottleneck_probability']:.3f}")
                print(f"    Drop: -{bottleneck['confidence_drop']:.3f} ({bottleneck['relative_drop_pct']:.1f}%)")
                print()
        
        if metrics['suspicious_spans']:
            print(f"  ⚠️  Suspicious Spans: {len(metrics['suspicious_spans'])}")
            for span in metrics['suspicious_spans']:
                print(f"     - {' '.join(span['tokens'])} (prob: {span['avg_probability']:.3f})")
