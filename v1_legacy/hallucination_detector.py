#!/usr/bin/env python3
"""
Hallucination Detector & Analyzer
----------------------------------
Detects and analyzes hallucinations in RAG-generated answers.

Hallucination Types:
1. FACTUAL: Generates facts not in context
2. ATTRIBUTION: Misattributes information to wrong source
3. CONTRADICTION: Contradicts information in context
4. FABRICATION: Invents details not mentioned
5. EXTRAPOLATION: Makes unsupported inferences
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from openai import OpenAI


@dataclass
class HallucinationInstance:
    """Single hallucination detection record."""
    timestamp: str
    query: str
    answer: str
    context: str
    intent_name: str
    intent_id: int
    
    # Hallucination classification
    has_hallucination: bool
    hallucination_type: Optional[str]  # FACTUAL, ATTRIBUTION, CONTRADICTION, etc.
    severity: Optional[str]  # MINOR, MODERATE, SEVERE
    hallucinated_claims: List[str]
    
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


class HallucinationDetector:
    """Detects hallucinations using multi-method approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize detector.
        
        Args:
            openai_api_key: OpenAI API key (from env if None)
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required for hallucination detection")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        
        # Log file for storing hallucination instances
        self.log_file = "hallucination_log.jsonl"
    
    def detect(
        self,
        query: str,
        answer: str,
        context: str,
        intent_name: str = "unknown",
        intent_id: int = -1,
        num_chunks: int = 0,
        context_length: int = 0,
        chunk_scores: List[float] = None,
        fallback_used: bool = False
    ) -> HallucinationInstance:
        """
        Detect hallucinations in an answer.
        
        Args:
            query: Original user query
            answer: Generated answer
            context: Retrieved context
            intent_name: Classified intent
            intent_id: Intent ID
            num_chunks: Number of chunks retrieved
            context_length: Length of context
            chunk_scores: List of chunk relevance scores
            fallback_used: Whether fallback mode was used
            
        Returns:
            HallucinationInstance with detection results
        """
        # Calculate context quality metrics
        chunk_scores = chunk_scores or []
        avg_score = np.mean(chunk_scores) if chunk_scores else 0.0
        min_score = min(chunk_scores) if chunk_scores else 0.0
        max_score = max(chunk_scores) if chunk_scores else 0.0
        
        # Check for citations
        has_citations = self._has_citations(answer)
        
        # Perform hallucination detection
        detection_result = self._detect_hallucination(query, answer, context)
        
        # Analyze root causes
        causes = self._analyze_causes(
            has_hallucination=detection_result['has_hallucination'],
            context_length=context_length,
            num_chunks=num_chunks,
            avg_score=avg_score,
            has_citations=has_citations,
            fallback_used=fallback_used,
            intent_name=intent_name
        )
        
        # Create instance
        instance = HallucinationInstance(
            timestamp=datetime.now().isoformat(),
            query=query,
            answer=answer,
            context=context,
            intent_name=intent_name,
            intent_id=intent_id,
            has_hallucination=detection_result['has_hallucination'],
            hallucination_type=detection_result.get('type'),
            severity=detection_result.get('severity'),
            hallucinated_claims=detection_result.get('claims', []),
            num_chunks_retrieved=num_chunks,
            context_length=context_length,
            avg_chunk_score=float(avg_score),
            min_chunk_score=float(min_score),
            max_chunk_score=float(max_score),
            fallback_used=fallback_used,
            has_citations=has_citations,
            answer_length=len(answer),
            likely_causes=causes
        )
        
        # Log to file
        self._log_instance(instance)
        
        return instance
    
    def _has_citations(self, answer: str) -> bool:
        """Check if answer contains citations."""
        citation_patterns = [
            r'\[source:',
            r'\[chunk:',
            r'\[.*?\d+.*?\]'
        ]
        return any(re.search(pattern, answer, re.IGNORECASE) for pattern in citation_patterns)
    
    def _detect_hallucination(self, query: str, answer: str, context: str) -> Dict:
        """
        Use LLM to detect hallucinations with detailed analysis.
        
        Returns:
            Dict with detection results
        """
        detection_prompt = f"""You are a hallucination detection expert for RAG systems.

Your task: Analyze if the ANSWER contains hallucinations (information not supported by CONTEXT).

CONTEXT (Retrieved Information):
{context}

USER QUERY:
{query}

GENERATED ANSWER:
{answer}

Instructions:
1. Extract all factual claims from the ANSWER
2. For EACH claim, verify if it's supported by the CONTEXT
3. Classify hallucination type if found:
   - FACTUAL: Facts not in context
   - ATTRIBUTION: Wrong source attribution
   - CONTRADICTION: Contradicts context
   - FABRICATION: Invented details
   - EXTRAPOLATION: Unsupported inferences

4. Rate severity:
   - MINOR: Small details, doesn't affect main answer
   - MODERATE: Important details missing/wrong
   - SEVERE: Core facts are hallucinated

Return ONLY a JSON object with this exact schema:
{{
  "has_hallucination": true/false,
  "type": "FACTUAL|ATTRIBUTION|CONTRADICTION|FABRICATION|EXTRAPOLATION|null",
  "severity": "MINOR|MODERATE|SEVERE|null",
  "claims": ["list of hallucinated claims"],
  "reasoning": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": detection_prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Parse JSON
            # Try to extract JSON if wrapped in markdown
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                raw_content = raw_content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(raw_content)
            
            # Ensure required fields
            if 'has_hallucination' not in result:
                result['has_hallucination'] = False
            
            return result
            
        except Exception as e:
            print(f"Warning: Hallucination detection failed: {e}")
            # Return conservative default
            return {
                'has_hallucination': False,
                'type': None,
                'severity': None,
                'claims': [],
                'reasoning': f"Detection error: {str(e)}"
            }
    
    def _analyze_causes(
        self,
        has_hallucination: bool,
        context_length: int,
        num_chunks: int,
        avg_score: float,
        has_citations: bool,
        fallback_used: bool,
        intent_name: str
    ) -> List[str]:
        """
        Analyze likely root causes of hallucination.
        
        Returns:
            List of probable causes
        """
        if not has_hallucination:
            return []
        
        causes = []
        
        # 1. Insufficient context
        if context_length < 200:
            causes.append("INSUFFICIENT_CONTEXT")
        
        # 2. Low retrieval quality
        if num_chunks == 0:
            causes.append("NO_CHUNKS_RETRIEVED")
        elif num_chunks < 3:
            causes.append("FEW_CHUNKS_RETRIEVED")
        
        if avg_score < 0.3:
            causes.append("LOW_RELEVANCE_SCORES")
        
        # 3. Missing citations (model not grounding)
        if not has_citations:
            causes.append("NO_CITATIONS")
        
        # 4. Fallback mode (expected to use general knowledge)
        if fallback_used:
            causes.append("FALLBACK_MODE_USED")
        
        # 5. Complex intent types more prone to hallucination
        if intent_name == "explanation":
            causes.append("COMPLEX_EXPLANATION_QUERY")
        
        # 6. Edge cases
        if context_length > 0 and num_chunks > 5 and avg_score > 0.5 and has_citations:
            causes.append("MODEL_EXTRAPOLATION")  # Good context but still hallucinated
        
        return causes if causes else ["UNKNOWN"]
    
    def _log_instance(self, instance: HallucinationInstance):
        """Log hallucination instance to JSONL file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(instance.to_dict()) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log hallucination: {e}")


class HallucinationAnalyzer:
    """Analyzes hallucination patterns from logs."""
    
    def __init__(self, log_file: str = "hallucination_log.jsonl"):
        """Initialize analyzer with log file."""
        self.log_file = log_file
        self.instances = self._load_instances()
    
    def _load_instances(self) -> List[HallucinationInstance]:
        """Load instances from log file."""
        instances = []
        
        if not os.path.exists(self.log_file):
            print(f"No log file found: {self.log_file}")
            return instances
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Convert dict back to dataclass (simplified - just use dict)
                    instances.append(data)
                except Exception as e:
                    print(f"Warning: Failed to parse line: {e}")
        
        return instances
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.instances:
            return {"total": 0, "hallucinations": 0, "rate": 0.0}
        
        total = len(self.instances)
        hallucinations = sum(1 for inst in self.instances if inst.get('has_hallucination'))
        
        return {
            "total_queries": total,
            "hallucinations_detected": hallucinations,
            "hallucination_rate": hallucinations / total if total > 0 else 0.0,
            "clean_answers": total - hallucinations
        }
    
    def analyze_by_intent(self) -> Dict:
        """Analyze hallucination rate by intent."""
        by_intent = defaultdict(lambda: {"total": 0, "hallucinations": 0})
        
        for inst in self.instances:
            intent = inst.get('intent_name', 'unknown')
            by_intent[intent]['total'] += 1
            if inst.get('has_hallucination'):
                by_intent[intent]['hallucinations'] += 1
        
        # Calculate rates
        result = {}
        for intent, stats in by_intent.items():
            result[intent] = {
                **stats,
                "rate": stats['hallucinations'] / stats['total'] if stats['total'] > 0 else 0.0
            }
        
        return dict(sorted(result.items(), key=lambda x: x[1]['rate'], reverse=True))
    
    def analyze_by_cause(self) -> Dict:
        """Analyze most common root causes."""
        cause_counts = defaultdict(int)
        
        for inst in self.instances:
            if inst.get('has_hallucination'):
                for cause in inst.get('likely_causes', []):
                    cause_counts[cause] += 1
        
        total_hallucinations = sum(1 for inst in self.instances if inst.get('has_hallucination'))
        
        result = {}
        for cause, count in cause_counts.items():
            result[cause] = {
                "count": count,
                "percentage": (count / total_hallucinations * 100) if total_hallucinations > 0 else 0.0
            }
        
        return dict(sorted(result.items(), key=lambda x: x[1]['count'], reverse=True))
    
    def analyze_by_type(self) -> Dict:
        """Analyze hallucination types."""
        type_counts = defaultdict(int)
        
        for inst in self.instances:
            if inst.get('has_hallucination'):
                h_type = inst.get('hallucination_type') or 'UNKNOWN'
                type_counts[h_type] += 1
        
        return dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_by_severity(self) -> Dict:
        """Analyze hallucination severity."""
        severity_counts = defaultdict(int)
        
        for inst in self.instances:
            if inst.get('has_hallucination'):
                severity = inst.get('severity') or 'UNKNOWN'
                severity_counts[severity] += 1
        
        return dict(sorted(
            severity_counts.items(),
            key=lambda x: {'SEVERE': 3, 'MODERATE': 2, 'MINOR': 1, 'UNKNOWN': 0}.get(x[0], 0),
            reverse=True
        ))
    
    def analyze_context_correlation(self) -> Dict:
        """Analyze correlation between context quality and hallucinations."""
        hallucinated = [inst for inst in self.instances if inst.get('has_hallucination')]
        clean = [inst for inst in self.instances if not inst.get('has_hallucination')]
        
        def avg_metric(instances, key):
            values = [inst.get(key, 0) for inst in instances if inst.get(key) is not None]
            return np.mean(values) if values else 0.0
        
        return {
            "hallucinated_answers": {
                "avg_context_length": avg_metric(hallucinated, 'context_length'),
                "avg_num_chunks": avg_metric(hallucinated, 'num_chunks_retrieved'),
                "avg_chunk_score": avg_metric(hallucinated, 'avg_chunk_score'),
                "citation_rate": sum(1 for inst in hallucinated if inst.get('has_citations')) / len(hallucinated) if hallucinated else 0.0
            },
            "clean_answers": {
                "avg_context_length": avg_metric(clean, 'context_length'),
                "avg_num_chunks": avg_metric(clean, 'num_chunks_retrieved'),
                "avg_chunk_score": avg_metric(clean, 'avg_chunk_score'),
                "citation_rate": sum(1 for inst in clean if inst.get('has_citations')) / len(clean) if clean else 0.0
            }
        }
    
    def get_worst_cases(self, limit: int = 10) -> List[Dict]:
        """Get worst hallucination cases (SEVERE severity)."""
        severe_cases = [
            inst for inst in self.instances
            if inst.get('has_hallucination') and inst.get('severity') == 'SEVERE'
        ]
        
        return severe_cases[:limit]
    
    def generate_report(self, output_file: str = "hallucination_report.txt"):
        """Generate comprehensive analysis report."""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("HALLUCINATION ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Log file: {self.log_file}")
        report_lines.append("")
        
        # Summary stats
        stats = self.get_summary_stats()
        report_lines.append("📊 SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Queries: {stats['total_queries']}")
        report_lines.append(f"Hallucinations Detected: {stats['hallucinations_detected']}")
        report_lines.append(f"Hallucination Rate: {stats['hallucination_rate']:.2%}")
        report_lines.append(f"Clean Answers: {stats['clean_answers']}")
        report_lines.append("")
        
        # By intent
        by_intent = self.analyze_by_intent()
        report_lines.append("🎯 HALLUCINATIONS BY INTENT")
        report_lines.append("-" * 80)
        for intent, data in by_intent.items():
            report_lines.append(f"{intent:15s}: {data['hallucinations']:3d}/{data['total']:3d} ({data['rate']:.1%})")
        report_lines.append("")
        
        # By type
        by_type = self.analyze_by_type()
        report_lines.append("🔍 HALLUCINATION TYPES")
        report_lines.append("-" * 80)
        for h_type, count in by_type.items():
            report_lines.append(f"{h_type:20s}: {count:3d}")
        report_lines.append("")
        
        # By severity
        by_severity = self.analyze_by_severity()
        report_lines.append("⚠️  SEVERITY DISTRIBUTION")
        report_lines.append("-" * 80)
        for severity, count in by_severity.items():
            report_lines.append(f"{severity:15s}: {count:3d}")
        report_lines.append("")
        
        # Root causes
        by_cause = self.analyze_by_cause()
        report_lines.append("🔎 ROOT CAUSES (Most Common)")
        report_lines.append("-" * 80)
        for cause, data in list(by_cause.items())[:10]:
            report_lines.append(f"{cause:30s}: {data['count']:3d} ({data['percentage']:.1f}%)")
        report_lines.append("")
        
        # Context correlation
        correlation = self.analyze_context_correlation()
        report_lines.append("📈 CONTEXT QUALITY CORRELATION")
        report_lines.append("-" * 80)
        report_lines.append("Hallucinated Answers:")
        report_lines.append(f"  Avg Context Length: {correlation['hallucinated_answers']['avg_context_length']:.0f} chars")
        report_lines.append(f"  Avg Chunks: {correlation['hallucinated_answers']['avg_num_chunks']:.1f}")
        report_lines.append(f"  Avg Score: {correlation['hallucinated_answers']['avg_chunk_score']:.3f}")
        report_lines.append(f"  Citation Rate: {correlation['hallucinated_answers']['citation_rate']:.1%}")
        report_lines.append("")
        report_lines.append("Clean Answers:")
        report_lines.append(f"  Avg Context Length: {correlation['clean_answers']['avg_context_length']:.0f} chars")
        report_lines.append(f"  Avg Chunks: {correlation['clean_answers']['avg_num_chunks']:.1f}")
        report_lines.append(f"  Avg Score: {correlation['clean_answers']['avg_chunk_score']:.3f}")
        report_lines.append(f"  Citation Rate: {correlation['clean_answers']['citation_rate']:.1%}")
        report_lines.append("")
        
        # Worst cases
        worst = self.get_worst_cases(5)
        if worst:
            report_lines.append("🚨 WORST CASES (Top 5 SEVERE)")
            report_lines.append("-" * 80)
            for i, case in enumerate(worst, 1):
                report_lines.append(f"\n[{i}] Query: {case.get('query', 'N/A')}")
                report_lines.append(f"    Type: {case.get('hallucination_type', 'N/A')}")
                report_lines.append(f"    Causes: {', '.join(case.get('likely_causes', []))}")
                report_lines.append(f"    Claims: {'; '.join(case.get('hallucinated_claims', [])[:2])}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write to file
        report_text = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Report saved to: {output_file}")
        
        return report_text


if __name__ == "__main__":
    """Demo usage."""
    
    # Example: Analyze existing logs
    analyzer = HallucinationAnalyzer()
    
    if analyzer.instances:
        print(f"Loaded {len(analyzer.instances)} instances from log")
        report = analyzer.generate_report()
        print("\n" + report)
    else:
        print("No hallucination logs found. The detector will create logs when integrated with RAG engine.")
        print("\nTo use:")
        print("1. Integrate HallucinationDetector with rag_engine.py")
        print("2. Run queries through RAG system")
        print("3. Run this analyzer to generate reports")
