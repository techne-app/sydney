#!/usr/bin/env python3
"""
Intent Detection Model Comparison Framework

Always evaluates all models and compares their performance on intent detection tasks.
Automatically saves results to mlc_llm/model_comparison.json for tracking progress.

Usage:
    python mlc_llm/eval_intent_detection.py                                    # Full evaluation
    python mlc_llm/eval_intent_detection.py --dataset pinned_thread_summary    # Test specific category  
    python mlc_llm/eval_intent_detection.py --temp 0.2 --quiet                 # Adjust temperature, reduce output
    python mlc_llm/eval_intent_detection.py --export-results custom.json       # Save to additional file
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from intent_evaluator import IntentEvaluator, EvalResult

# Global output file path
OUTPUT_JSON_PATH = "mlc_llm/model_comparison.json"

@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    size_mb: float
    avg_inference_time: float  # Average time per inference
    total_eval_time: float
    total_test_cases: int

# Test cases now loaded from JSON file

DEFAULT_MODELS = [
    "Phi-3.5-mini-instruct-q4f16_1-MLC",
    "gemma-2-2b-it-q4f16_1-MLC", 
    "Llama-3.2-3B-Instruct-q4f16_1-MLC"
]


def extract_detailed_analysis(model_name: str, eval_results: List) -> dict:
    """Extract detailed failure and success analysis from evaluation results, grouped by iterations"""
    test_cases_analysis = []
    
    for eval_result in eval_results:
        # Group iterations for this test case
        iterations_data = []
        
        for i, individual_result in enumerate(eval_result.individual_results, 1):
            iteration_info = {
                "iteration_number": i,
                "predicted": individual_result.predicted,
                "confidence": individual_result.confidence,
                "reasoning": individual_result.reasoning,
                "raw_response": individual_result.raw_response,
                "correct": individual_result.correct,
                "inference_time": individual_result.inference_time
            }
            iterations_data.append(iteration_info)
        
        # Separate successes and failures for summary
        successes = [iter_data for iter_data in iterations_data if iter_data["correct"]]
        failures = [iter_data for iter_data in iterations_data if not iter_data["correct"]]
        
        test_case_info = {
            "test_case_id": getattr(eval_result, 'id', f"unknown_{hash(eval_result.question) % 10000}"),
            "question": eval_result.question,
            "category": eval_result.category,
            "difficulty": eval_result.difficulty,
            "expected": eval_result.intent_expected,
            "notes": eval_result.notes,
            
            # Summary metrics
            "accuracy": eval_result.accuracy,
            "iterations_total": eval_result.iterations,
            "iterations_successful": len(successes),
            "iterations_failed": len(failures),
            "most_common_prediction": eval_result.most_common_prediction,
            "prediction_consistency": eval_result.prediction_consistency,
            "mean_confidence": eval_result.mean_confidence,
            "mean_confidence_when_right": eval_result.mean_confidence_correct if eval_result.mean_confidence_correct else 0,
            "mean_confidence_when_wrong": eval_result.mean_confidence_incorrect if eval_result.mean_confidence_incorrect else 0,
            
            # Detailed iteration data
            "iterations": iterations_data,
            
            # Quick access to successes/failures
            "successful_iterations": successes,
            "failed_iterations": failures
        }
        
        test_cases_analysis.append(test_case_info)
    
    # Calculate aggregate statistics
    all_failures = []
    all_successes = []
    
    for case in test_cases_analysis:
        if case["iterations_failed"] > 0:
            all_failures.append({
                "category": case["category"],
                "difficulty": case["difficulty"]
            })
        if case["iterations_successful"] > 0:
            all_successes.append({
                "category": case["category"], 
                "difficulty": case["difficulty"]
            })
    
    return {
        "total_test_cases": len(test_cases_analysis),
        "total_cases_with_failures": sum(1 for case in test_cases_analysis if case["iterations_failed"] > 0),
        "total_cases_with_successes": sum(1 for case in test_cases_analysis if case["iterations_successful"] > 0),
        "failure_rate_by_category": _calculate_failure_by_category(all_failures, all_successes),
        "failure_rate_by_difficulty": _calculate_failure_by_difficulty(all_failures, all_successes),
        "test_cases": test_cases_analysis
    }


def _calculate_failure_by_category(failures, successes):
    """Calculate failure rates grouped by category"""
    from collections import defaultdict
    category_stats = defaultdict(lambda: {"total": 0, "failed": 0})
    
    for failure in failures:
        category_stats[failure["category"]]["total"] += 1
        category_stats[failure["category"]]["failed"] += 1
    
    for success in successes:
        category_stats[success["category"]]["total"] += 1
    
    return {cat: {"failure_rate": stats["failed"] / stats["total"], **stats} 
            for cat, stats in category_stats.items()}


def _calculate_failure_by_difficulty(failures, successes):
    """Calculate failure rates grouped by difficulty"""
    from collections import defaultdict
    difficulty_stats = defaultdict(lambda: {"total": 0, "failed": 0})
    
    for failure in failures:
        difficulty_stats[failure["difficulty"]]["total"] += 1
        difficulty_stats[failure["difficulty"]]["failed"] += 1
    
    for success in successes:
        difficulty_stats[success["difficulty"]]["total"] += 1
    
    return {diff: {"failure_rate": stats["failed"] / stats["total"], **stats} 
            for diff, stats in difficulty_stats.items()}


def evaluate_all_models(temperature: float = 0.1, dataset_filter: str = None, iterations: int = 10, verbose: bool = False) -> List[ModelMetrics]:
    """Evaluate all default models and return comprehensive metrics"""
    print(f"🚀 EVALUATING ALL MODELS: {', '.join(DEFAULT_MODELS)}")
    print("=" * 80)
    
    model_metrics = []
    
    for i, model_name in enumerate(DEFAULT_MODELS, 1):
        print(f"\n📊 [{i}/{len(DEFAULT_MODELS)}] Evaluating {model_name}...")
        print(f"🚀 Loading {model_name}...")
        
        try:
            evaluator = IntentEvaluator(model_name)
            results = evaluator.run_evaluation(iterations=iterations, temperature=temperature, dataset_filter=dataset_filter, verbose=verbose)
            metrics = evaluator.calculate_aggregated_metrics(results)
            
            # Create ModelMetrics object
            model_metric = ModelMetrics(
                name=model_name,
                accuracy=metrics.get('overall_accuracy', 0),
                size_mb=evaluator.model_size_mb,
                avg_inference_time=metrics.get('mean_inference_time', 0),
                total_eval_time=evaluator.total_eval_time,
                total_test_cases=metrics.get('total_test_cases', 0)
            )
            
            model_metrics.append(model_metric)
            
            # Store detailed results for failure analysis
            if not hasattr(evaluate_all_models, '_detailed_results'):
                evaluate_all_models._detailed_results = {}
            evaluate_all_models._detailed_results[model_name] = results
            
            # Quick summary for this model
            print(f"✅ {model_name} complete:")
            print(f"   Accuracy: {model_metric.accuracy:.1%}")
            print(f"   Size: {model_metric.size_mb:.1f} MB")
            print(f"   Avg Response: {model_metric.avg_inference_time:.3f}s")
            
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
    print(f"{'Model':<35} {'Size (MB)':<10} {'Accuracy':<10} {'Avg Time':<10} {'Status':<15}")
    print("-" * 90)
    
    # Table rows
    for metrics in sorted_metrics:
        # Determine status
        if metrics.accuracy >= 0.85:
            status = "✅ EXCELLENT"
        elif metrics.accuracy >= 0.80:
            status = "✅ GOOD"
        elif metrics.accuracy >= 0.75:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ POOR"
        
        model_name = metrics.name.replace("-q4f16_1-MLC", "")
        print(f"{model_name:<35} {metrics.size_mb:<10.1f} {metrics.accuracy:<10.1%} {metrics.avg_inference_time:<10.3f} {status:<15}")
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    accuracies = [m.accuracy for m in model_metrics]
    sizes = [m.size_mb for m in model_metrics]
    times = [m.avg_inference_time for m in model_metrics]
    
    print(f"   Best Accuracy: {max(accuracies):.1%} ({sorted_metrics[0].name.replace('-q4f16_1-MLC', '')})")
    print(f"   Smallest Model: {min(sizes):.1f} MB")
    print(f"   Fastest Inference: {min(times):.3f}s")
    
    # Smart recommendations considering multiple factors
    # Find models with best accuracy
    max_accuracy = max(m.accuracy for m in model_metrics)
    top_accuracy_models = [m for m in model_metrics if m.accuracy == max_accuracy]
    
    # Among top accuracy models, find the best overall (fastest + smallest)
    if len(top_accuracy_models) > 1:
        # Use combined score: lower is better (normalized speed + size)
        def combined_score(model):
            # Normalize to 0-1 range
            speed_norm = model.avg_inference_time / max(m.avg_inference_time for m in top_accuracy_models)
            size_norm = model.size_mb / max(m.size_mb for m in top_accuracy_models)
            return speed_norm + size_norm  # Equal weight to speed and size
        
        best_overall = min(top_accuracy_models, key=combined_score)
    else:
        best_overall = top_accuracy_models[0]
    
    smallest = min(model_metrics, key=lambda x: x.size_mb)
    fastest = min(model_metrics, key=lambda x: x.avg_inference_time)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🏆 Best Overall: {best_overall.name.replace('-q4f16_1-MLC', '')} ({best_overall.accuracy:.1%} accuracy, {best_overall.avg_inference_time:.3f}s, {best_overall.size_mb:.0f}MB)")
    print(f"   📱 Most Efficient: {smallest.name.replace('-q4f16_1-MLC', '')} ({smallest.size_mb:.1f} MB)")
    print(f"   ⚡ Fastest: {fastest.name.replace('-q4f16_1-MLC', '')} ({fastest.avg_inference_time:.3f}s avg)")




def main():
    parser = argparse.ArgumentParser(description='Comprehensive intent detection model comparison - always evaluates all models')
    parser.add_argument('--dataset', help='Filter test cases by category (e.g., "pinned_thread_summary", "ambiguous")')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per test case for statistical analysis (default: 10)')
    parser.add_argument('--export-results', help='Export results to additional custom file (.json)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Always evaluate all models (this is the only meaningful evaluation)
    try:
        model_metrics = evaluate_all_models(
            temperature=args.temp,
            dataset_filter=args.dataset,
            iterations=args.iterations,
            verbose=not args.quiet
        )
        
        print_model_comparison_table(model_metrics)
        
        # Always export comprehensive results to model_comparison.json
        export_data = {
            "metadata": {
                "total_models": len(model_metrics),
                "test_cases": model_metrics[0].total_test_cases if model_metrics else 0,
                "temperature": args.temp,
                "dataset_filter": args.dataset,
                "iterations": args.iterations,
                "evaluation_timestamp": __import__('datetime').datetime.now().isoformat()
            },
            "models": []
        }
        
        # Add detailed analysis for each model
        for model_metric in model_metrics:
            model_data = asdict(model_metric)
            
            # Add detailed failure and success analysis if available
            if hasattr(evaluate_all_models, '_detailed_results') and model_metric.name in evaluate_all_models._detailed_results:
                detailed_analysis = extract_detailed_analysis(
                    model_metric.name, 
                    evaluate_all_models._detailed_results[model_metric.name]
                )
                model_data['detailed_analysis'] = detailed_analysis
            
            export_data["models"].append(model_data)
        
        # Always write to default location
        default_filepath = Path(OUTPUT_JSON_PATH)
        with open(default_filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\n📄 Results automatically saved to {default_filepath}")
        
        # Also export to custom location if specified
        if args.export_results:
            custom_filepath = Path(args.export_results)
            with open(custom_filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📄 Results also exported to {custom_filepath}")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
