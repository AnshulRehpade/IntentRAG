#!/usr/bin/env python3
"""
Run Hallucination Analysis
--------------------------
Analyzes hallucination patterns and generates comprehensive reports
"""

import argparse
from hallucination_detector import HallucinationAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze hallucination patterns in RAG system')
    
    parser.add_argument('--log-file', type=str, default='hallucination_log.jsonl',
                        help='Path to hallucination log file (default: hallucination_log.jsonl)')
    parser.add_argument('--output', type=str, default='hallucination_report.txt',
                        help='Output report file (default: hallucination_report.txt)')
    parser.add_argument('--format', type=str, choices=['text', 'json'], default='text',
                        help='Output format (default: text)')
    
    args = parser.parse_args()
    
    print(f"Loading hallucination logs from: {args.log_file}")
    analyzer = HallucinationAnalyzer(log_file=args.log_file)
    
    if not analyzer.instances:
        print("❌ No hallucination logs found!")
        print("\nTo collect hallucination data:")
        print("1. Run queries through the RAG system (streamlit_app.py or rag_engine.py)")
        print("2. Hallucinations will be automatically detected and logged")
        print("3. Run this script again to analyze patterns")
        return
    
    print(f"✅ Loaded {len(analyzer.instances)} query instances")
    
    # Generate report
    if args.format == 'text':
        report = analyzer.generate_report(output_file=args.output)
        print("\n" + report)
    else:
        # JSON format
        import json
        
        report_data = {
            "summary": analyzer.get_summary_stats(),
            "by_intent": analyzer.analyze_by_intent(),
            "by_cause": analyzer.analyze_by_cause(),
            "by_type": analyzer.analyze_by_type(),
            "by_severity": analyzer.analyze_by_severity(),
            "context_correlation": analyzer.analyze_context_correlation(),
            "worst_cases": analyzer.get_worst_cases(10)
        }
        
        with open(args.output, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✅ JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
