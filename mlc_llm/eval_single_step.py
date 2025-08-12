#!/usr/bin/env python3
"""
Single-Step Intent Detection Model Evaluation

Evaluates models using the single-step approach that combines:
1. Intent classification (action vs chat)
2. Function selection (if action)

Provides direct comparison against two-step approach for performance analysis.

Usage:
    python mlc_llm/eval_single_step.py                                           # Full evaluation
    python mlc_llm/eval_single_step.py --dataset pinned_thread_summary           # Test specific category
    python mlc_llm/eval_single_step.py --temp 0.2 --iterations 5 --quiet         # Adjust settings
    python mlc_llm/eval_single_step.py --export-results single_step_results.json # Save to custom file
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import statistics

from single_step_evaluator import SingleStepEvaluator, SingleStepAggregatedResult

# Global output file path
OUTPUT_JSON_PATH = "mlc_llm/single_step_comparison.json"

@dataclass
class SingleStepModelMetrics:
    name: str
    size_mb: float
    
    # Single-step metrics
    intent_accuracy: float
    function_accuracy: float
    overall_accuracy: float
    mean_confidence: float
    mean_inference_time: float
    
    # Evaluation metadata
    total_eval_time: float
    total_test_cases: int

DEFAULT_MODELS = [
    "Phi-3.5-mini-instruct-q4f16_1-MLC",
    "gemma-2-2b-it-q4f16_1-MLC", 
    "Llama-3.2-3B-Instruct-q4f16_1-MLC"
]

def evaluate_all_models_single_step(temperature: float = 0.1, dataset_filter: str = None, iterations: int = 10, verbose: bool = False) -> List[SingleStepModelMetrics]:
    """Evaluate all default models using single-step approach"""
    print(f"🚀 SINGLE-STEP EVALUATION: {', '.join(DEFAULT_MODELS)}")
    print("=" * 80)
    print("📋 Combined Intent Detection + Function Calling in One Step")
    print("=" * 80)
    
    model_metrics = []
    
    for i, model_name in enumerate(DEFAULT_MODELS, 1):
        print(f"\\n📊 [{i}/{len(DEFAULT_MODELS)}] Evaluating {model_name}...")
        print(f"🚀 Loading {model_name}...")
        
        try:
            evaluator = SingleStepEvaluator(model_name)
            results = evaluator.run_evaluation(iterations=iterations, temperature=temperature, dataset_filter=dataset_filter, verbose=verbose)
            metrics = evaluator.calculate_aggregated_metrics(results)
            
            # Create SingleStepModelMetrics object
            model_metric = SingleStepModelMetrics(
                name=model_name,
                size_mb=evaluator.model_size_mb,
                intent_accuracy=metrics.get('intent_overall_accuracy', 0),
                function_accuracy=metrics.get('function_overall_accuracy', 0),
                overall_accuracy=metrics.get('overall_accuracy', 0),
                mean_confidence=metrics.get('mean_confidence', 0),
                mean_inference_time=metrics.get('mean_inference_time', 0),
                total_eval_time=evaluator.total_eval_time,
                total_test_cases=metrics.get('total_test_cases', 0)
            )
            
            model_metrics.append(model_metric)
            
            # Store detailed results for analysis
            if not hasattr(evaluate_all_models_single_step, '_detailed_results'):
                evaluate_all_models_single_step._detailed_results = {}
            evaluate_all_models_single_step._detailed_results[model_name] = results
            
            # Quick summary for this model
            print(f"✅ {model_name} complete:")
            print(f"   Intent Accuracy: {model_metric.intent_accuracy:.1%}")
            print(f"   Function Accuracy: {model_metric.function_accuracy:.1%}")
            print(f"   Overall Accuracy: {model_metric.overall_accuracy:.1%}")
            print(f"   Size: {model_metric.size_mb:.1f} MB")
            print(f"   Avg Inference Time: {model_metric.mean_inference_time:.3f}s")
            
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

def print_single_step_comparison_table(model_metrics: List[SingleStepModelMetrics]) -> None:
    """Print comprehensive single-step comparison table"""
    if not model_metrics:
        print("❌ No model metrics to display")
        return
    
    print(f"\\n🏆 SINGLE-STEP MODEL COMPARISON TABLE")
    print("=" * 110)
    
    # Sort by overall accuracy (descending)
    sorted_metrics = sorted(model_metrics, key=lambda x: x.overall_accuracy, reverse=True)
    
    # Table header
    print(f"{'Model':<25} {'Size':<8} {'Intent Acc':<11} {'Function Acc':<13} {'Overall':<9} {'Avg Time':<10} {'Status':<15}")
    print("-" * 108)
    
    # Table rows
    for metrics in sorted_metrics:
        # Determine status
        if metrics.overall_accuracy >= 0.85:
            status = "✅ EXCELLENT"
        elif metrics.overall_accuracy >= 0.80:
            status = "✅ GOOD"
        elif metrics.overall_accuracy >= 0.75:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ POOR"
        
        model_name = metrics.name.replace("-q4f16_1-MLC", "")
        
        print(f"{model_name:<25} {metrics.size_mb:<8.0f} {metrics.intent_accuracy:<11.1%} {metrics.function_accuracy:<13.1%} {metrics.overall_accuracy:<9.1%} {metrics.mean_inference_time:<10.3f} {status:<15}")
    
    # Summary statistics
    print(f"\\n📈 SINGLE-STEP SUMMARY STATISTICS:")
    intent_accuracies = [m.intent_accuracy for m in model_metrics]
    function_accuracies = [m.function_accuracy for m in model_metrics]
    overall_accuracies = [m.overall_accuracy for m in model_metrics]
    inference_times = [m.mean_inference_time for m in model_metrics]
    
    print(f"   Best Intent Accuracy: {max(intent_accuracies):.1%}")
    print(f"   Best Function Accuracy: {max(function_accuracies):.1%}")
    print(f"   Best Overall Accuracy: {max(overall_accuracies):.1%}")
    print(f"   Average Intent: {statistics.mean(intent_accuracies):.1%}")
    print(f"   Average Function: {statistics.mean(function_accuracies):.1%}")
    print(f"   Average Overall: {statistics.mean(overall_accuracies):.1%}")
    print(f"   Fastest Inference: {min(inference_times):.3f}s")
    print(f"   Average Inference: {statistics.mean(inference_times):.3f}s")
    
    # Find best performing model
    best_overall = sorted_metrics[0]
    best_intent = max(model_metrics, key=lambda x: x.intent_accuracy)
    best_function = max(model_metrics, key=lambda x: x.function_accuracy)
    fastest_model = min(model_metrics, key=lambda x: x.mean_inference_time)
    
    print(f"\\n💡 SINGLE-STEP RECOMMENDATIONS:")
    print(f"   🏆 Best Overall: {best_overall.name.replace('-q4f16_1-MLC', '')} ({best_overall.overall_accuracy:.1%} accuracy)")
    print(f"   🎯 Best Intent: {best_intent.name.replace('-q4f16_1-MLC', '')} ({best_intent.intent_accuracy:.1%} accuracy)")
    print(f"   ⚙️  Best Function: {best_function.name.replace('-q4f16_1-MLC', '')} ({best_function.function_accuracy:.1%} accuracy)")
    print(f"   ⚡ Fastest: {fastest_model.name.replace('-q4f16_1-MLC', '')} ({fastest_model.mean_inference_time:.3f}s avg)")

def extract_single_step_detailed_analysis(model_name: str, eval_results: List[SingleStepAggregatedResult]) -> dict:
    """Extract detailed analysis from single-step evaluation results"""
    test_cases_analysis = []
    
    for result in eval_results:
        test_case_info = {
            "test_case_id": getattr(result, 'id', f"single_step_{hash(result.question) % 10000}"),
            "question": result.question,
            "category": result.category,
            "difficulty": result.difficulty,
            "intent_expected": result.intent_expected,
            "function_expected": result.function_expected,
            "notes": result.notes,
            
            # Aggregated metrics
            "intent_accuracy": result.intent_accuracy,
            "function_accuracy": result.function_accuracy,
            "overall_accuracy": result.overall_accuracy,
            "mean_confidence": result.mean_confidence,
            "mean_inference_time": result.mean_inference_time,
            "iterations_total": result.iterations,
            
            # Individual iteration details
            "iterations": [
                {
                    "iteration_number": i + 1,
                    "intent_predicted": iter_result.intent_predicted,
                    "function_predicted": iter_result.function_predicted,
                    "confidence": iter_result.confidence,
                    "reasoning": iter_result.reasoning,
                    "inference_time": iter_result.inference_time,
                    "intent_correct": iter_result.intent_correct,
                    "function_correct": iter_result.function_correct,
                    "overall_correct": iter_result.overall_correct,
                    "raw_response": iter_result.raw_response
                }
                for i, iter_result in enumerate(result.individual_results)
            ]
        }
        
        test_cases_analysis.append(test_case_info)
    
    return {
        "total_test_cases": len(test_cases_analysis),
        "test_cases": test_cases_analysis
    }

def main():
    parser = argparse.ArgumentParser(description='Single-step intent detection model evaluation')
    parser.add_argument('--dataset', help='Filter test cases by category (e.g., "pinned_thread_summary", "ambiguous")')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per test case (default: 10)')
    parser.add_argument('--export-results', help='Export results to additional custom file (.json)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        model_metrics = evaluate_all_models_single_step(
            temperature=args.temp,
            dataset_filter=args.dataset,
            iterations=args.iterations,
            verbose=not args.quiet
        )
        
        print_single_step_comparison_table(model_metrics)
        
        # Export comprehensive results
        export_data = {
            "metadata": {
                "evaluation_type": "single_step",
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
            
            # Add detailed analysis if available
            if hasattr(evaluate_all_models_single_step, '_detailed_results') and model_metric.name in evaluate_all_models_single_step._detailed_results:
                detailed_analysis = extract_single_step_detailed_analysis(
                    model_metric.name, 
                    evaluate_all_models_single_step._detailed_results[model_metric.name]
                )
                model_data['detailed_analysis'] = detailed_analysis
            
            export_data["models"].append(model_data)
        
        # Always write to default location
        default_filepath = Path(OUTPUT_JSON_PATH)
        with open(default_filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\\n📄 Single-step results saved to {default_filepath}")
        
        # Also export to custom location if specified
        if args.export_results:
            custom_filepath = Path(args.export_results)
            with open(custom_filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📄 Single-step results also exported to {custom_filepath}")
            
    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error in single-step evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())