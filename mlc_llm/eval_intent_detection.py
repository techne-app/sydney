#!/usr/bin/env python3
"""
Intent Detection Evaluation Framework

Comprehensive evaluation of intent detection models for chat vs search classification.
Builds on quick-intent-test.py with extensive test cases and evaluation metrics.

Usage:
    python mlc_llm/eval_intent_detection.py --full-eval
    python mlc_llm/eval_intent_detection.py --dataset ambiguous --temp 0.2
    python mlc_llm/eval_intent_detection.py --analyze-failures
    python mlc_llm/eval_intent_detection.py --export-results results.json
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from intent_evaluator import IntentEvaluator, AggregatedEvalResult



@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    size_mb: float
    avg_response_time: float
    total_eval_time: float
    total_test_cases: int

# Test cases now loaded from JSON file

DEFAULT_MODELS = [
    "Phi-3.5-mini-instruct-q4f16_1-MLC",
    "gemma-2-2b-it-q4f16_1-MLC", 
    "Llama-3.2-3B-Instruct-q4f16_1-MLC"
]


def evaluate_all_models(temperature: float = 0.1, dataset_filter: str = None, verbose: bool = False) -> List[ModelMetrics]:
    """Evaluate all default models and return comprehensive metrics"""
    print(f"🚀 EVALUATING ALL MODELS: {', '.join(DEFAULT_MODELS)}")
    print("=" * 80)
    
    model_metrics = []
    
    for i, model_name in enumerate(DEFAULT_MODELS, 1):
        print(f"\n📊 [{i}/{len(DEFAULT_MODELS)}] Evaluating {model_name}...")
        print(f"🚀 Loading {model_name}...")
        
        try:
            evaluator = IntentEvaluator(model_name)
            results = evaluator.run_full_evaluation(temperature, dataset_filter, verbose=verbose)
            metrics = evaluator.calculate_metrics(results)
            
            # Create ModelMetrics object
            model_metric = ModelMetrics(
                name=model_name,
                accuracy=metrics.get('accuracy', 0),
                precision=metrics.get('precision', 0),
                recall=metrics.get('recall', 0),
                f1_score=metrics.get('f1_score', 0),
                size_mb=evaluator.model_size_mb,
                avg_response_time=metrics.get('performance_stats', {}).get('avg_response_time', 0),
                total_eval_time=evaluator.total_eval_time,
                total_test_cases=len(results)
            )
            
            model_metrics.append(model_metric)
            
            # Quick summary for this model
            print(f"✅ {model_name} complete:")
            print(f"   Accuracy: {model_metric.accuracy:.1%}")
            print(f"   Size: {model_metric.size_mb:.1f} MB")
            print(f"   Avg Response: {model_metric.avg_response_time:.3f}s")
            
            # Clean up memory
            evaluator.cleanup()
            del evaluator
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            continue
    
    return model_metrics

def print_model_comparison_table(model_metrics: List[ModelMetrics]) -> None:
    """Print a comprehensive comparison table of all models"""
    if not model_metrics:
        print("❌ No model metrics to display")
        return
    
    print(f"\n🏆 MODEL COMPARISON TABLE")
    print("=" * 100)
    
    # Sort by accuracy (descending)
    sorted_metrics = sorted(model_metrics, key=lambda x: x.accuracy, reverse=True)
    
    # Table header
    print(f"{'Model':<35} {'Size (MB)':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<6} {'Avg Time':<10} {'Status':<15}")
    print("-" * 100)
    
    # Table rows
    for i, metrics in enumerate(sorted_metrics):
        # Determine status
        if metrics.accuracy >= 0.85:
            status = "✅ EXCELLENT"
        elif metrics.accuracy >= 0.80:
            status = "✅ GOOD"
        elif metrics.accuracy >= 0.75:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ POOR"
        
        # Highlight best model
        model_name = metrics.name.replace("-q4f16_1-MLC", "")
        if i == 0:
            model_name = f"🥇 {model_name}"
        elif i == 1:
            model_name = f"🥈 {model_name}"
        elif i == 2:
            model_name = f"🥉 {model_name}"
        
        print(f"{model_name:<35} {metrics.size_mb:<10.1f} {metrics.accuracy:<10.1%} "
              f"{metrics.precision:<11.1%} {metrics.recall:<8.1%} {metrics.f1_score:<6.3f} "
              f"{metrics.avg_response_time:<10.3f} {status:<15}")
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    accuracies = [m.accuracy for m in model_metrics]
    sizes = [m.size_mb for m in model_metrics]
    times = [m.avg_response_time for m in model_metrics]
    
    print(f"   Best Accuracy: {max(accuracies):.1%} ({sorted_metrics[0].name.replace('-q4f16_1-MLC', '')})")
    print(f"   Smallest Model: {min(sizes):.1f} MB")
    print(f"   Fastest Response: {min(times):.3f}s")
    
    # Recommendations
    best_overall = sorted_metrics[0]
    smallest = min(model_metrics, key=lambda x: x.size_mb)
    fastest = min(model_metrics, key=lambda x: x.avg_response_time)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🏆 Best Overall: {best_overall.name.replace('-q4f16_1-MLC', '')} ({best_overall.accuracy:.1%} accuracy)")
    print(f"   📱 Most Efficient: {smallest.name.replace('-q4f16_1-MLC', '')} ({smallest.size_mb:.1f} MB)")
    print(f"   ⚡ Fastest: {fastest.name.replace('-q4f16_1-MLC', '')} ({fastest.avg_response_time:.3f}s avg)")




def main():
    parser = argparse.ArgumentParser(description='Comprehensive intent detection evaluation')
    parser.add_argument('--full-eval', action='store_true', help='Run full evaluation on all test cases')
    parser.add_argument('--eval-all', action='store_true', help='Evaluate all default models (Phi, Gemma, Llama)')
    parser.add_argument('--dataset', help='Filter test cases by category (e.g., "ambiguous", "edge_case")')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--analyze-failures', action='store_true', help='Show detailed failure analysis')
    parser.add_argument('--export-results', help='Export results to file (.json or .csv)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--model', help=f'Model to use (default: {DEFAULT_MODELS[0]})')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per test case for statistical analysis (default: 10)')
    
    args = parser.parse_args()
    
    # Handle evaluation of all models
    if args.eval_all:
        try:
            model_metrics = evaluate_all_models(
                temperature=args.temp,
                dataset_filter=args.dataset,
                verbose=not args.quiet
            )
            
            print_model_comparison_table(model_metrics)
            
            if args.export_results:
                # Export comprehensive results
                export_data = {
                    "metadata": {
                        "total_models": len(model_metrics),
                        "test_cases": model_metrics[0].total_test_cases if model_metrics else 0,
                        "temperature": args.temp,
                        "dataset_filter": args.dataset
                    },
                    "models": [asdict(m) for m in model_metrics]
                }
                
                filepath = Path(args.export_results)
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                print(f"\n📄 Results exported to {filepath}")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in eval-all: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    
    # Single model evaluation
    if not any([args.full_eval, args.dataset, args.analyze_failures]):
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python mlc_llm/eval_intent_detection.py --eval-all")
        print(f"  python mlc_llm/eval_intent_detection.py --full-eval")
        print(f"  python mlc_llm/eval_intent_detection.py --full-eval --iterations 5  # 5 iterations per test case")
        print(f"  python mlc_llm/eval_intent_detection.py --dataset ambiguous --temp 0.2 --iterations 20")
        print(f"  python mlc_llm/eval_intent_detection.py --full-eval --export-results results.json")
        return
    
    model_name = args.model or DEFAULT_MODELS[0]
    evaluator = IntentEvaluator(model_name)
    
    try:
        if args.full_eval or args.dataset:
            if args.iterations > 1:
                # Run multi-iteration evaluation
                aggregated_results: List[AggregatedEvalResult] = evaluator.run_full_evaluation_with_iterations(
                    iterations=args.iterations,
                    temperature=args.temp,
                    dataset_filter=args.dataset,
                    verbose=not args.quiet
                )
                
                # Print aggregated report
                evaluator.print_aggregated_report(aggregated_results)
                
                if args.analyze_failures:
                    # Analyze failures from individual results
                    all_individual_results = []
                    for agg_result in aggregated_results:
                        all_individual_results.extend(agg_result.individual_results)
                    evaluator.analyze_failures(all_individual_results)
                    
                if args.export_results:
                    # Export aggregated results
                    export_data = {
                        "metadata": {
                            "evaluation_type": "multi_iteration",
                            "iterations_per_case": args.iterations,
                            "temperature": args.temp,
                            "dataset_filter": args.dataset,
                            "model_name": evaluator.model_name
                        },
                        "aggregated_metrics": evaluator.calculate_aggregated_metrics(aggregated_results),
                        "results": [
                            {
                                "question": r.question,
                                "intent_expected": r.intent_expected,
                                "category": r.category,
                                "difficulty": r.difficulty,
                                "accuracy": r.accuracy,
                                "mean_confidence": r.mean_confidence,
                                "confidence_std": r.confidence_std,
                                "most_common_prediction": r.most_common_prediction,
                                "prediction_consistency": r.prediction_consistency,
                                "iterations": r.iterations
                            } for r in aggregated_results
                        ]
                    }
                    
                    filepath = Path(args.export_results)
                    with open(filepath, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    print(f"\n📄 Multi-iteration results exported to {filepath}")
                    
            else:
                # Run single-iteration evaluation (original behavior)
                results = evaluator.run_full_evaluation(
                    temperature=args.temp,
                    dataset_filter=args.dataset,
                    verbose=not args.quiet
                )
                
                # Print comprehensive report
                evaluator.print_comprehensive_report(results)
                
                if args.analyze_failures:
                    evaluator.analyze_failures(results)
                    
                if args.export_results:
                    evaluator.export_results(args.export_results, results)
        
        elif args.analyze_failures:
            print("❌ No results to analyze. Run --full-eval first.")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error in main(): {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())