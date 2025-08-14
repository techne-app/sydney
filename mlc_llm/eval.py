#!/usr/bin/env python3
"""
Unified Intent Detection Model Evaluation

This script provides a unified interface for evaluating intent detection models
using either single-step, two-step, or both approaches for direct comparison.

Usage:
    python mlc_llm/eval.py                                    # Run both evaluations (default)
    python mlc_llm/eval.py --mode single                      # Single-step only
    python mlc_llm/eval.py --mode two                         # Two-step only
    python mlc_llm/eval.py --mode both                        # Both approaches
    python mlc_llm/eval.py --temp 0.2 --iterations 5          # Custom settings
    python mlc_llm/eval.py --dataset pinned_thread_summary    # Filter dataset
    python mlc_llm/eval.py --quiet                            # Reduce verbosity
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# Import evaluation modules
from eval_single_step import evaluate_all_models_single_step, print_single_step_comparison_table, extract_single_step_detailed_analysis, SingleStepModelMetrics
from eval_two_step import evaluate_all_models_two_step, print_two_step_comparison_table, extract_two_step_detailed_analysis, TwoStepModelMetrics

# Constants
PAST_EVALS_DIR = Path("mlc_llm/past_evals")
DEFAULT_MODELS = [
    "Phi-3.5-mini-instruct-q4f16_1-MLC",
    "gemma-2-2b-it-q4f16_1-MLC", 
    "Llama-3.2-3B-Instruct-q4f16_1-MLC"
]

@dataclass
class ComparisonMetrics:
    """Combined metrics for comparing single-step vs two-step approaches"""
    model_name: str
    single_step_accuracy: float
    two_step_accuracy: float
    accuracy_difference: float  # two_step - single_step
    single_step_time: float
    two_step_time: float
    time_difference: float  # two_step - single_step
    recommended_approach: str  # "single", "two", or "similar"

def generate_timestamp() -> str:
    """Generate timestamp string for filenames"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_timestamped_filename(evaluation_type: str, timestamp: str) -> str:
    """Create timestamped filename for evaluation results"""
    return f"{evaluation_type}_{timestamp}.json"

def save_evaluation_results(evaluation_type: str, results_data: dict, timestamp: str) -> Path:
    """Save evaluation results to timestamped file in past_evals directory"""
    # Ensure past_evals directory exists
    PAST_EVALS_DIR.mkdir(exist_ok=True)
    
    filename = create_timestamped_filename(evaluation_type, timestamp)
    filepath = PAST_EVALS_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return filepath

def run_single_step_evaluation(temperature: float, dataset_filter: Optional[str], 
                              iterations: int, verbose: bool, timestamp: str) -> Tuple[List[SingleStepModelMetrics], Path]:
    """Run single-step evaluation and save results"""
    print("🔍 RUNNING SINGLE-STEP EVALUATION")
    print("=" * 50)
    
    model_metrics = evaluate_all_models_single_step(
        temperature=temperature,
        dataset_filter=dataset_filter,
        iterations=iterations,
        verbose=verbose
    )
    
    print_single_step_comparison_table(model_metrics)
    
    # Prepare results for saving
    export_data = {
        "metadata": {
            "evaluation_type": "single_step",
            "total_models": len(model_metrics),
            "test_cases": model_metrics[0].total_test_cases if model_metrics else 0,
            "temperature": temperature,
            "dataset_filter": dataset_filter,
            "iterations": iterations,
            "evaluation_timestamp": datetime.now().isoformat()
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
    
    # Save results
    filepath = save_evaluation_results("single_step", export_data, timestamp)
    print(f"\n📄 Single-step results saved to {filepath}")
    
    return model_metrics, filepath

def run_two_step_evaluation(temperature: float, dataset_filter: Optional[str], 
                           iterations: int, verbose: bool, timestamp: str) -> Tuple[List[TwoStepModelMetrics], Path]:
    """Run two-step evaluation and save results"""
    print("🔍 RUNNING TWO-STEP EVALUATION")
    print("=" * 50)
    
    model_metrics = evaluate_all_models_two_step(
        temperature=temperature,
        dataset_filter=dataset_filter,
        iterations=iterations,
        verbose=verbose
    )
    
    print_two_step_comparison_table(model_metrics)
    
    # Prepare results for saving
    export_data = {
        "metadata": {
            "evaluation_type": "two_step",
            "total_models": len(model_metrics),
            "test_cases": model_metrics[0].total_test_cases if model_metrics else 0,
            "temperature": temperature,
            "dataset_filter": dataset_filter,
            "iterations": iterations,
            "evaluation_timestamp": datetime.now().isoformat()
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
    
    # Save results
    filepath = save_evaluation_results("two_step", export_data, timestamp)
    print(f"\n📄 Two-step results saved to {filepath}")
    
    return model_metrics, filepath

def generate_comparison_analysis(single_step_metrics: List[SingleStepModelMetrics], 
                               two_step_metrics: List[TwoStepModelMetrics]) -> List[ComparisonMetrics]:
    """Generate comparative analysis between single-step and two-step approaches"""
    
    # Create lookup dictionary for two-step metrics
    two_step_lookup = {m.name: m for m in two_step_metrics}
    
    comparison_metrics = []
    
    for single_metric in single_step_metrics:
        model_name = single_metric.name
        
        if model_name in two_step_lookup:
            two_metric = two_step_lookup[model_name]
            
            # Calculate differences
            accuracy_diff = two_metric.overall_accuracy - single_metric.overall_accuracy
            time_diff = two_metric.avg_total_time - single_metric.mean_inference_time
            
            # Determine recommended approach
            if abs(accuracy_diff) < 0.02:  # Less than 2% difference
                recommended = "similar"
            elif accuracy_diff > 0.02:
                recommended = "two"
            else:
                recommended = "single"
            
            comparison = ComparisonMetrics(
                model_name=model_name,
                single_step_accuracy=single_metric.overall_accuracy,
                two_step_accuracy=two_metric.overall_accuracy,
                accuracy_difference=accuracy_diff,
                single_step_time=single_metric.mean_inference_time,
                two_step_time=two_metric.avg_total_time,
                time_difference=time_diff,
                recommended_approach=recommended
            )
            
            comparison_metrics.append(comparison)
    
    return comparison_metrics

def print_comparison_table(comparison_metrics: List[ComparisonMetrics]) -> None:
    """Print comprehensive comparison table between single-step and two-step approaches"""
    if not comparison_metrics:
        print("❌ No comparison metrics to display")
        return
    
    print(f"\n🏆 SINGLE-STEP vs TWO-STEP COMPARISON")
    print("=" * 130)
    
    # Sort by accuracy difference (descending)
    sorted_metrics = sorted(comparison_metrics, key=lambda x: x.accuracy_difference, reverse=True)
    
    # Table header
    print(f"{'Model':<25} {'Single Acc':<11} {'Two Acc':<9} {'Acc Diff':<10} {'Single Time':<12} {'Two Time':<10} {'Time Diff':<11} {'Recommended':<12}")
    print("-" * 128)
    
    # Table rows
    for metrics in sorted_metrics:
        model_name = metrics.model_name.replace("-q4f16_1-MLC", "")
        
        # Format accuracy difference with sign
        acc_diff_str = f"{metrics.accuracy_difference:+.1%}"
        time_diff_str = f"{metrics.time_difference:+.3f}s"
        
        # Color code recommendation
        if metrics.recommended_approach == "two":
            rec_display = "🔵 Two-Step"
        elif metrics.recommended_approach == "single":
            rec_display = "🟡 Single-Step"
        else:
            rec_display = "⚪ Similar"
        
        print(f"{model_name:<25} {metrics.single_step_accuracy:<11.1%} {metrics.two_step_accuracy:<9.1%} {acc_diff_str:<10} {metrics.single_step_time:<12.3f} {metrics.two_step_time:<10.3f} {time_diff_str:<11} {rec_display:<12}")
    
    # Summary statistics
    print(f"\n📊 COMPARISON SUMMARY:")
    accuracy_improvements = [m.accuracy_difference for m in comparison_metrics if m.accuracy_difference > 0]
    time_increases = [m.time_difference for m in comparison_metrics]
    
    two_step_better = len([m for m in comparison_metrics if m.recommended_approach == "two"])
    single_step_better = len([m for m in comparison_metrics if m.recommended_approach == "single"])
    similar_performance = len([m for m in comparison_metrics if m.recommended_approach == "similar"])
    
    print(f"   Models favoring two-step: {two_step_better}")
    print(f"   Models favoring single-step: {single_step_better}")
    print(f"   Models with similar performance: {similar_performance}")
    
    if accuracy_improvements:
        print(f"   Average accuracy improvement (two-step): {statistics.mean(accuracy_improvements):.1%}")
        print(f"   Max accuracy improvement: {max(accuracy_improvements):.1%}")
    
    print(f"   Average time increase (two-step): {statistics.mean(time_increases):.3f}s")
    
    # Best performing model for each approach
    best_single = max(comparison_metrics, key=lambda x: x.single_step_accuracy)
    best_two = max(comparison_metrics, key=lambda x: x.two_step_accuracy)
    
    print(f"\n🏅 BEST PERFORMERS:")
    print(f"   Single-step: {best_single.model_name.replace('-q4f16_1-MLC', '')} ({best_single.single_step_accuracy:.1%})")
    print(f"   Two-step: {best_two.model_name.replace('-q4f16_1-MLC', '')} ({best_two.two_step_accuracy:.1%})")

def save_comparison_results(comparison_metrics: List[ComparisonMetrics], 
                          single_path: Path, two_path: Path, timestamp: str) -> Path:
    """Save comparison analysis results"""
    comparison_data = {
        "metadata": {
            "evaluation_type": "comparison",
            "single_step_results_file": str(single_path),
            "two_step_results_file": str(two_path),
            "total_models": len(comparison_metrics),
            "evaluation_timestamp": datetime.now().isoformat()
        },
        "comparisons": [asdict(metric) for metric in comparison_metrics],
        "summary": {
            "two_step_better_count": len([m for m in comparison_metrics if m.recommended_approach == "two"]),
            "single_step_better_count": len([m for m in comparison_metrics if m.recommended_approach == "single"]),
            "similar_performance_count": len([m for m in comparison_metrics if m.recommended_approach == "similar"]),
            "average_accuracy_difference": statistics.mean([m.accuracy_difference for m in comparison_metrics]),
            "average_time_difference": statistics.mean([m.time_difference for m in comparison_metrics])
        }
    }
    
    filepath = save_evaluation_results("comparison", comparison_data, timestamp)
    print(f"\n📄 Comparison analysis saved to {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Unified intent detection model evaluation')
    parser.add_argument('--mode', choices=['single', 'two', 'both'], default='both', 
                       help='Evaluation mode: single-step, two-step, or both (default: both)')
    parser.add_argument('--dataset', help='Filter test cases by category (e.g., "pinned_thread_summary", "ambiguous")')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations per test case (default: 10)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Generate timestamp for this evaluation session
    timestamp = generate_timestamp()
    
    print("🚀 UNIFIED INTENT DETECTION EVALUATION")
    print("=" * 60)
    print(f"📅 Session: {timestamp}")
    print(f"🎯 Mode: {args.mode.upper()}")
    print(f"🌡️  Temperature: {args.temp}")
    print(f"🔄 Iterations: {args.iterations}")
    if args.dataset:
        print(f"📊 Dataset Filter: {args.dataset}")
    print("=" * 60)
    
    try:
        single_step_metrics = None
        two_step_metrics = None
        single_path = None
        two_path = None
        
        # Run evaluations based on mode
        if args.mode in ['single', 'both']:
            single_step_metrics, single_path = run_single_step_evaluation(
                temperature=args.temp,
                dataset_filter=args.dataset,
                iterations=args.iterations,
                verbose=not args.quiet,
                timestamp=timestamp
            )
        
        if args.mode in ['two', 'both']:
            two_step_metrics, two_path = run_two_step_evaluation(
                temperature=args.temp,
                dataset_filter=args.dataset,
                iterations=args.iterations,
                verbose=not args.quiet,
                timestamp=timestamp
            )
        
        # Generate comparison if both evaluations were run
        if args.mode == 'both' and single_step_metrics and two_step_metrics:
            print("\n" + "=" * 60)
            comparison_metrics = generate_comparison_analysis(single_step_metrics, two_step_metrics)
            print_comparison_table(comparison_metrics)
            save_comparison_results(comparison_metrics, single_path, two_path, timestamp)
        
        print(f"\n✅ Evaluation complete! Results stored in {PAST_EVALS_DIR}")
        
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