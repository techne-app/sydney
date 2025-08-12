#!/usr/bin/env python3
"""
Two-Step Intent Detection Model Evaluation

Evaluates models using the two-step approach:
1. Step 1: Intent classification (action vs chat)
2. Step 2: Function selection (for action cases only)

Provides separate accuracy metrics for each step plus combined accuracy.

Usage:
    python mlc_llm/eval_two_step.py                                           # Full evaluation
    python mlc_llm/eval_two_step.py --dataset pinned_thread_summary           # Test specific category
    python mlc_llm/eval_two_step.py --temp 0.2 --iterations 5 --quiet         # Adjust settings
    python mlc_llm/eval_two_step.py --export-results two_step_results.json    # Save to custom file
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import statistics

from two_step_evaluator import TwoStepEvaluator, TwoStepAggregatedResult

# Global output file path
OUTPUT_JSON_PATH = "mlc_llm/two_step_comparison.json"

@dataclass
class TwoStepModelMetrics:
    name: str
    size_mb: float
    
    # Step 1 metrics
    step1_accuracy: float
    step1_avg_time: float
    
    # Step 2 metrics  
    step2_accuracy: float  # None if no step2 cases
    step2_avg_time: float
    step2_cases: int  # Number of cases that reached step 2
    
    # Combined metrics
    overall_accuracy: float
    avg_total_time: float
    total_eval_time: float
    total_test_cases: int

DEFAULT_MODELS = [
    "Phi-3.5-mini-instruct-q4f16_1-MLC",
    "gemma-2-2b-it-q4f16_1-MLC", 
    "Llama-3.2-3B-Instruct-q4f16_1-MLC"
]

def evaluate_all_models_two_step(temperature: float = 0.1, dataset_filter: str = None, iterations: int = 10, verbose: bool = False) -> List[TwoStepModelMetrics]:
    """Evaluate all default models using two-step approach"""
    print(f"🚀 TWO-STEP EVALUATION: {', '.join(DEFAULT_MODELS)}")
    print("=" * 80)
    print("📋 Step 1: Intent Detection (action vs chat)")
    print("📋 Step 2: Function Selection (when intent=action)")
    print("=" * 80)
    
    model_metrics = []
    
    for i, model_name in enumerate(DEFAULT_MODELS, 1):
        print(f"\\n📊 [{i}/{len(DEFAULT_MODELS)}] Evaluating {model_name}...")
        print(f"🚀 Loading {model_name}...")
        
        try:
            evaluator = TwoStepEvaluator(model_name)
            results = evaluator.run_evaluation(iterations=iterations, temperature=temperature, dataset_filter=dataset_filter, verbose=verbose)
            metrics = evaluator.calculate_aggregated_metrics(results)
            
            # Create TwoStepModelMetrics object
            model_metric = TwoStepModelMetrics(
                name=model_name,
                size_mb=evaluator.model_size_mb,
                step1_accuracy=metrics.get('step1_overall_accuracy', 0),
                step1_avg_time=statistics.mean([r.step1_mean_time for r in results]) if results else 0,
                step2_accuracy=metrics.get('step2_overall_accuracy', 0) or 0,
                step2_avg_time=statistics.mean([r.step2_mean_time for r in results if r.step2_mean_time]) if results else 0,
                step2_cases=metrics.get('step2_applicable_cases', 0),
                overall_accuracy=metrics.get('overall_accuracy', 0),
                avg_total_time=metrics.get('mean_inference_time', 0),
                total_eval_time=evaluator.total_eval_time,
                total_test_cases=metrics.get('total_test_cases', 0)
            )
            
            model_metrics.append(model_metric)
            
            # Store detailed results for analysis
            if not hasattr(evaluate_all_models_two_step, '_detailed_results'):
                evaluate_all_models_two_step._detailed_results = {}
            evaluate_all_models_two_step._detailed_results[model_name] = results
            
            # Quick summary for this model
            print(f"✅ {model_name} complete:")
            print(f"   Step 1 Accuracy: {model_metric.step1_accuracy:.1%}")
            print(f"   Step 2 Accuracy: {model_metric.step2_accuracy:.1%} ({model_metric.step2_cases} cases)")
            print(f"   Overall Accuracy: {model_metric.overall_accuracy:.1%}")
            print(f"   Size: {model_metric.size_mb:.1f} MB")
            print(f"   Avg Total Time: {model_metric.avg_total_time:.3f}s")
            
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

def print_two_step_comparison_table(model_metrics: List[TwoStepModelMetrics]) -> None:
    """Print comprehensive two-step comparison table"""
    if not model_metrics:
        print("❌ No model metrics to display")
        return
    
    print(f"\\n🏆 TWO-STEP MODEL COMPARISON TABLE")
    print("=" * 120)
    
    # Sort by overall accuracy (descending)
    sorted_metrics = sorted(model_metrics, key=lambda x: x.overall_accuracy, reverse=True)
    
    # Table header
    print(f"{'Model':<25} {'Size':<8} {'Step1 Acc':<10} {'Step2 Acc':<10} {'Overall':<9} {'Step1 Time':<11} {'Step2 Time':<11} {'Status':<15}")
    print("-" * 118)
    
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
        step2_acc_str = f"{metrics.step2_accuracy:.1%}" if metrics.step2_cases > 0 else "N/A"
        step2_time_str = f"{metrics.step2_avg_time:.3f}s" if metrics.step2_cases > 0 else "N/A"
        
        print(f"{model_name:<25} {metrics.size_mb:<8.0f} {metrics.step1_accuracy:<10.1%} {step2_acc_str:<10} {metrics.overall_accuracy:<9.1%} {metrics.step1_avg_time:<11.3f} {step2_time_str:<11} {status:<15}")
    
    # Summary statistics
    print(f"\\n📈 TWO-STEP SUMMARY STATISTICS:")
    step1_accuracies = [m.step1_accuracy for m in model_metrics]
    step2_accuracies = [m.step2_accuracy for m in model_metrics if m.step2_cases > 0]
    overall_accuracies = [m.overall_accuracy for m in model_metrics]
    
    print(f"   Best Step 1 Accuracy: {max(step1_accuracies):.1%}")
    if step2_accuracies:
        print(f"   Best Step 2 Accuracy: {max(step2_accuracies):.1%}")
    print(f"   Best Overall Accuracy: {max(overall_accuracies):.1%}")
    print(f"   Average Step 1: {statistics.mean(step1_accuracies):.1%}")
    if step2_accuracies:
        print(f"   Average Step 2: {statistics.mean(step2_accuracies):.1%}")
    print(f"   Average Overall: {statistics.mean(overall_accuracies):.1%}")
    
    # Find best performing model
    best_overall = sorted_metrics[0]
    best_step1 = max(model_metrics, key=lambda x: x.step1_accuracy)
    best_step2 = max([m for m in model_metrics if m.step2_cases > 0], key=lambda x: x.step2_accuracy) if step2_accuracies else None
    
    print(f"\\n💡 TWO-STEP RECOMMENDATIONS:")
    print(f"   🏆 Best Overall: {best_overall.name.replace('-q4f16_1-MLC', '')} ({best_overall.overall_accuracy:.1%} combined accuracy)")
    print(f"   🎯 Best Step 1 (Intent): {best_step1.name.replace('-q4f16_1-MLC', '')} ({best_step1.step1_accuracy:.1%} accuracy)")
    if best_step2:
        print(f"   ⚙️  Best Step 2 (Function): {best_step2.name.replace('-q4f16_1-MLC', '')} ({best_step2.step2_accuracy:.1%} accuracy)")

def extract_two_step_detailed_analysis(model_name: str, eval_results: List[TwoStepAggregatedResult]) -> dict:
    """Extract detailed analysis from two-step evaluation results"""
    test_cases_analysis = []
    
    for result in eval_results:
        test_case_info = {
            "test_case_id": getattr(result, 'id', f"two_step_{hash(result.question) % 10000}"),
            "question": result.question,
            "category": result.category,
            "difficulty": result.difficulty,
            "intent_expected": result.intent_expected,
            "function_expected": result.function_expected,
            "notes": result.notes,
            
            # Step 1 metrics
            "step1_accuracy": result.step1_accuracy,
            "step1_mean_confidence": result.step1_mean_confidence,
            "step1_mean_time": result.step1_mean_time,
            
            # Step 2 metrics
            "step2_accuracy": result.step2_accuracy,
            "step2_mean_confidence": result.step2_mean_confidence,
            "step2_mean_time": result.step2_mean_time,
            "step2_cases_count": result.step2_cases_count,
            
            # Overall metrics
            "overall_accuracy": result.overall_accuracy,
            "mean_total_time": result.mean_total_time,
            "mean_final_confidence": result.mean_final_confidence,
            "iterations_total": result.iterations,
            
            # Individual iteration details
            "iterations": [
                {
                    "iteration_number": i + 1,
                    "step1_predicted": iter_result.step1_predicted,
                    "step1_confidence": iter_result.step1_confidence,
                    "step1_correct": iter_result.step1_correct,
                    "step1_reasoning": iter_result.step1_reasoning,
                    "step1_time": iter_result.step1_time,
                    "step2_predicted": iter_result.step2_predicted,
                    "step2_confidence": iter_result.step2_confidence,
                    "step2_correct": iter_result.step2_correct,
                    "step2_reasoning": iter_result.step2_reasoning,
                    "step2_time": iter_result.step2_time,
                    "overall_correct": iter_result.overall_correct,
                    "total_time": iter_result.total_time,
                    "final_confidence": iter_result.final_confidence
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
    parser = argparse.ArgumentParser(description='Two-step intent detection model evaluation')
    parser.add_argument('--dataset', help='Filter test cases by category (e.g., "pinned_thread_summary", "ambiguous")')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per test case (default: 10)')
    parser.add_argument('--export-results', help='Export results to additional custom file (.json)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        model_metrics = evaluate_all_models_two_step(
            temperature=args.temp,
            dataset_filter=args.dataset,
            iterations=args.iterations,
            verbose=not args.quiet
        )
        
        print_two_step_comparison_table(model_metrics)
        
        # Export comprehensive results
        export_data = {
            "metadata": {
                "evaluation_type": "two_step",
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
            if hasattr(evaluate_all_models_two_step, '_detailed_results') and model_metric.name in evaluate_all_models_two_step._detailed_results:
                detailed_analysis = extract_two_step_detailed_analysis(
                    model_metric.name, 
                    evaluate_all_models_two_step._detailed_results[model_metric.name]
                )
                model_data['detailed_analysis'] = detailed_analysis
            
            export_data["models"].append(model_data)
        
        # Always write to default location
        default_filepath = Path(OUTPUT_JSON_PATH)
        with open(default_filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\\n📄 Two-step results saved to {default_filepath}")
        
        # Also export to custom location if specified
        if args.export_results:
            custom_filepath = Path(args.export_results)
            with open(custom_filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📄 Two-step results also exported to {custom_filepath}")
            
    except KeyboardInterrupt:
        print("\\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error in two-step evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())