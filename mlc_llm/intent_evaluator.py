#!/usr/bin/env python3
"""
Intent Evaluator Class

Core evaluation functionality for intent detection models.
Handles test case evaluation, metrics calculation, and reporting.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics
import math
import time

from tqdm import tqdm
from model_wrapper import MLCModelWrapper
from testcase_loader import TestCaseLoader, TestCase

@dataclass
class SingleEvalResult:
    question: str
    intent_expected: str
    predicted: str
    confidence: float
    reasoning: str
    raw_response: str
    correct: bool
    category: str
    difficulty: str
    notes: str
    inference_time: float  # Time in seconds for this inference
    model_name: str = ""

@dataclass
class EvalResult:
    """Aggregated results across multiple iterations per test case"""
    question: str
    intent_expected: str
    category: str
    difficulty: str
    notes: str
    model_name: str
    
    # Aggregated metrics
    iterations: int
    accuracy: float  # Percentage of correct predictions across iterations
    mean_confidence: float
    confidence_std: float
    mean_confidence_correct: float  # Mean confidence when prediction was correct
    mean_confidence_incorrect: float  # Mean confidence when prediction was wrong
    
    # Timing metrics
    mean_inference_time: float  # Average inference time across iterations
    inference_time_std: float   # Standard deviation of inference times
    
    # Most common prediction (mode)
    most_common_prediction: str
    prediction_consistency: float  # Percentage of iterations with most common prediction
    
    # Raw iteration data
    individual_results: List[SingleEvalResult]

def load_prompt_from_typescript():
    """Load the shared prompt from TypeScript file"""
    try:
        script_dir = Path(__file__).parent
        ts_file = script_dir.parent / "src" / "prompts" / "searchIntent.ts"
        
        if not ts_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {ts_file}")
        
        content = ts_file.read_text()
        match = re.search(r'export const SEARCH_INTENT_PROMPT = `(.*?)`;', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find SEARCH_INTENT_PROMPT in TypeScript file")
        
        prompt_template = match.group(1).strip()
        return prompt_template
        
    except Exception as e:
        print(f"❌ Failed to load prompt from TypeScript: {e}")
        print("❌ Cannot proceed without the actual prompt from searchIntent.ts")
        print("❌ Please ensure src/prompts/searchIntent.ts exists and contains SEARCH_INTENT_PROMPT")
        sys.exit(1)

class IntentEvaluator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "Phi-3.5-mini-instruct-q4f16_1-MLC"
        self.model_wrapper = MLCModelWrapper(self.model_name)
        self.prompt_template = load_prompt_from_typescript()
        self.results: List[SingleEvalResult] = []
        self.total_eval_time = 0.0
    
    @property
    def model_size_mb(self) -> float:
        """Get model size from wrapper"""
        return self.model_wrapper.model_size_mb
    
    def parse_response(self, response: str) -> Dict:
        """Parse JSON response from model using wrapper's parser"""
        parsed = self.model_wrapper.parse_json_response(response)
        
        if "error" in parsed:
            return parsed
            
        # Validate intent detection specific fields
        if "isSearch" not in parsed or not isinstance(parsed["isSearch"], bool):
            return {"error": "Invalid isSearch field", "raw": response}
        
        if "confidence" not in parsed or not (0 <= parsed["confidence"] <= 1):
            return {"error": "Invalid confidence field", "raw": response}
        
        # Convert to standard format
        parsed["intentCategory"] = "action" if parsed["isSearch"] else "chat"
        return parsed
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model_wrapper:
            self.model_wrapper.cleanup()
    
    def evaluate_test_case(self, test_case: TestCase, iterations: int = 10, temperature: float = 0.1, verbose: bool = True) -> Optional[EvalResult]:
        """Evaluate a single test case multiple times and aggregate results"""
        if verbose:
            if iterations == 1:
                print(f"\n🧪 Testing: \"{test_case.question}\" ({test_case.category}, {test_case.difficulty})")
            else:
                print(f"\n🧪 Testing: \"{test_case.question}\" ({test_case.category}, {test_case.difficulty}) - {iterations} iterations")
        
        individual_results = []
        
        # Run multiple iterations
        for iteration in range(iterations):
            if verbose and iterations > 1:
                print(f"   Iteration {iteration + 1}/{iterations}", end="... ")
            
            result = self._evaluate_single_case(test_case, temperature)
            if result:
                individual_results.append(result)
            
            if verbose and iterations > 1:
                status = "✅" if result and result.correct else "❌"
                print(f"{status} {result.predicted if result else 'ERROR'} ({result.confidence:.2f})" if result else "ERROR")
        
        if not individual_results:
            return None
        
        # Calculate aggregated metrics
        correct_count = sum(1 for r in individual_results if r.correct)
        accuracy = correct_count / len(individual_results)
        
        confidences = [r.confidence for r in individual_results]
        mean_confidence = statistics.mean(confidences)
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        correct_confidences = [r.confidence for r in individual_results if r.correct]
        incorrect_confidences = [r.confidence for r in individual_results if not r.correct]
        
        mean_confidence_correct = statistics.mean(correct_confidences) if correct_confidences else 0.0
        mean_confidence_incorrect = statistics.mean(incorrect_confidences) if incorrect_confidences else 0.0
        
        # Calculate timing metrics
        inference_times = [r.inference_time for r in individual_results]
        mean_inference_time = statistics.mean(inference_times) if inference_times else 0.0
        inference_time_std = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        
        # Find most common prediction
        predictions = [r.predicted for r in individual_results]
        prediction_counts = Counter(predictions)
        most_common_prediction = prediction_counts.most_common(1)[0][0]
        prediction_consistency = prediction_counts[most_common_prediction] / len(predictions)
        
        aggregated_result = EvalResult(
            question=test_case.question,
            intent_expected=test_case.intent_expected,
            category=test_case.category,
            difficulty=test_case.difficulty,
            notes=test_case.notes,
            model_name=self.model_name,
            iterations=len(individual_results),
            accuracy=accuracy,
            mean_confidence=mean_confidence,
            confidence_std=confidence_std,
            mean_confidence_correct=mean_confidence_correct,
            mean_confidence_incorrect=mean_confidence_incorrect,
            mean_inference_time=mean_inference_time,
            inference_time_std=inference_time_std,
            most_common_prediction=most_common_prediction,
            prediction_consistency=prediction_consistency,
            individual_results=individual_results
        )
        
        if verbose:
            status = "✅ MOSTLY CORRECT" if accuracy >= 0.5 else "❌ MOSTLY WRONG"
            print(f"   📊 Aggregated: {accuracy:.1%} accuracy, {prediction_consistency:.1%} consistency → {status}")
            if accuracy < 1.0 and accuracy > 0.0:
                print(f"   🎯 Most common: {most_common_prediction} ({prediction_consistency:.1%}), Expected: {test_case.intent_expected}")
        
        return aggregated_result
    
    def _evaluate_single_case(self, test_case: TestCase, temperature: float = 0.1) -> Optional[SingleEvalResult]:
        """Internal method to evaluate a single iteration of a test case"""
        # Build prompt
        prompt = self.prompt_template.replace('{message}', test_case.question)
        
        # Measure inference time
        start_time = time.time()
        response = self.model_wrapper.call_model(prompt, temperature, max_tokens=500)
        inference_time = time.time() - start_time
        
        if not response:
            return None
        
        # Parse response
        parsed = self.parse_response(response)
        
        if "error" in parsed:
            return None
        
        # Create result
        predicted = parsed.get('intentCategory', 'ERROR')
        correct = predicted == test_case.intent_expected
        
        result = SingleEvalResult(
            question=test_case.question,
            intent_expected=test_case.intent_expected,
            predicted=predicted,
            confidence=parsed.get('confidence', 0.0),
            reasoning=parsed.get('reasoning', ''),
            raw_response=response,
            correct=correct,
            category=test_case.category,
            difficulty=test_case.difficulty,
            notes=test_case.notes,
            inference_time=inference_time,
            model_name=self.model_name
        )
        
        return result
    
    def run_evaluation(self, iterations: int = 10, temperature: float = 0.1, dataset_filter: str = None, verbose: bool = True) -> List[EvalResult]:
        """Run evaluation with multiple iterations per test case"""
        loader = TestCaseLoader("mlc_llm/bfcl_testcases.json")
        
        if dataset_filter:
            test_cases = loader.get_cases_by_category(dataset_filter)
            print(f"📊 Running {iterations}-iteration evaluation on {len(test_cases)} test cases (filter: {dataset_filter})")
        else:
            test_cases = loader.test_cases
            print(f"📊 Running {iterations}-iteration evaluation on {len(test_cases)} test cases")
        
        print(f"🚀 Loading {self.model_name}...")
        print(f"🔄 Each test case will be run {iterations} times for statistical analysis")
        
        results = []
        total_iterations = len(test_cases) * iterations
        completed_iterations = 0
        
        # Use tqdm for progress tracking
        if not verbose:
            iterator = tqdm(test_cases, desc=f"Evaluating {self.model_name} ({iterations}x)", unit="test")
        else:
            iterator = test_cases
        
        for i, test_case in enumerate(iterator, 1):
            if verbose:
                print(f"\nProgress: {i}/{len(test_cases)} (completed {completed_iterations}/{total_iterations} total iterations)")
            
            result = self.evaluate_test_case(test_case, iterations, temperature, verbose)
            if result:
                results.append(result)
                completed_iterations += result.iterations
        
        return results
    
    
    def calculate_metrics(self, results: List[SingleEvalResult] = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        # Basic metrics
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        # Confidence analysis
        confidences = [r.confidence for r in results]
        correct_confidences = [r.confidence for r in results if r.correct]
        incorrect_confidences = [r.confidence for r in results if not r.correct]
        
        return {
            "accuracy": accuracy,
            "total_cases": total,
            "correct": correct,
            "confidence_stats": {
                "mean_confidence": statistics.mean(confidences) if confidences else 0,
                "mean_correct_confidence": statistics.mean(correct_confidences) if correct_confidences else 0,
                "mean_incorrect_confidence": statistics.mean(incorrect_confidences) if incorrect_confidences else 0,
                "confidence_stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0
            }
        }
    
    def analyze_by_category(self, results: List[SingleEvalResult] = None) -> Dict:
        """Analyze results by test case category"""
        if results is None:
            results = self.results
        
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "cases": []})
        
        for result in results:
            category_stats[result.category]["total"] += 1
            if result.correct:
                category_stats[result.category]["correct"] += 1
            category_stats[result.category]["cases"].append(result)
        
        # Calculate accuracy per category
        for category in category_stats:
            stats = category_stats[category]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        return dict(category_stats)
    
    def analyze_by_difficulty(self, results: List[SingleEvalResult] = None) -> Dict:
        """Analyze results by difficulty level"""
        if results is None:
            results = self.results
        
        difficulty_stats = defaultdict(lambda: {"correct": 0, "total": 0, "cases": []})
        
        for result in results:
            difficulty_stats[result.difficulty]["total"] += 1
            if result.correct:
                difficulty_stats[result.difficulty]["correct"] += 1
            difficulty_stats[result.difficulty]["cases"].append(result)
        
        # Calculate accuracy per difficulty
        for difficulty in difficulty_stats:
            stats = difficulty_stats[difficulty]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        return dict(difficulty_stats)
    
    def calculate_aggregated_metrics(self, aggregated_results: List[EvalResult]) -> Dict:
        """Calculate metrics from aggregated multi-iteration results"""
        if not aggregated_results:
            return {}
        
        # Overall accuracy (mean across all test cases)
        overall_accuracy = statistics.mean([r.accuracy for r in aggregated_results])
        accuracy_std = statistics.stdev([r.accuracy for r in aggregated_results]) if len(aggregated_results) > 1 else 0.0
        
        # Confidence metrics
        mean_confidences = [r.mean_confidence for r in aggregated_results]
        overall_mean_confidence = statistics.mean(mean_confidences)
        
        # Consistency metrics
        consistency_scores = [r.prediction_consistency for r in aggregated_results]
        mean_consistency = statistics.mean(consistency_scores)
        consistency_std = statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0
        
        # Count test cases by accuracy thresholds
        perfect_cases = sum(1 for r in aggregated_results if r.accuracy == 1.0)
        mostly_correct = sum(1 for r in aggregated_results if r.accuracy >= 0.5)
        
        # Calculate confidence intervals for overall accuracy
        n = len(aggregated_results)
        margin_of_error = 1.96 * (accuracy_std / math.sqrt(n)) if n > 1 else 0.0  # 95% CI
        confidence_interval = (overall_accuracy - margin_of_error, overall_accuracy + margin_of_error)
        
        # Timing metrics
        inference_times = [r.mean_inference_time for r in aggregated_results]
        overall_mean_inference_time = statistics.mean(inference_times)
        inference_time_std = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        
        return {
            "overall_accuracy": overall_accuracy,
            "accuracy_std": accuracy_std,
            "confidence_interval_95": confidence_interval,
            "total_test_cases": len(aggregated_results),
            "perfect_cases": perfect_cases,
            "mostly_correct_cases": mostly_correct,
            "mean_confidence": overall_mean_confidence,
            "mean_consistency": mean_consistency,
            "consistency_std": consistency_std,
            "mean_inference_time": overall_mean_inference_time,
            "inference_time_std": inference_time_std,
            "total_iterations": sum(r.iterations for r in aggregated_results)
        }
    
    def print_aggregated_report(self, aggregated_results: List[EvalResult]) -> None:
        """Print comprehensive report for multi-iteration evaluation"""
        if not aggregated_results:
            print("❌ No results to report")
            return
        
        metrics = self.calculate_aggregated_metrics(aggregated_results)
        
        print(f"\n📊 MULTI-ITERATION EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n🎯 STATISTICAL PERFORMANCE:")
        print(f"   Mean Accuracy: {metrics['overall_accuracy']:.1%} ± {metrics['accuracy_std']:.1%}")
        ci_low, ci_high = metrics['confidence_interval_95']
        print(f"   95% Confidence Interval: [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"   Perfect Cases: {metrics['perfect_cases']}/{metrics['total_test_cases']} ({metrics['perfect_cases']/metrics['total_test_cases']:.1%})")
        print(f"   Mostly Correct (≥50%): {metrics['mostly_correct_cases']}/{metrics['total_test_cases']} ({metrics['mostly_correct_cases']/metrics['total_test_cases']:.1%})")
        
        print(f"\n🎲 CONSISTENCY ANALYSIS:")
        print(f"   Mean Prediction Consistency: {metrics['mean_consistency']:.1%} ± {metrics['consistency_std']:.1%}")
        print(f"   Total Iterations Completed: {metrics['total_iterations']:,}")
        print(f"   Mean Confidence: {metrics['mean_confidence']:.3f}")
        
        print(f"\n⏱️ PERFORMANCE ANALYSIS:")
        print(f"   Mean Inference Time: {metrics['mean_inference_time']:.3f}s ± {metrics['inference_time_std']:.3f}s")
        print(f"   Total Evaluation Time: {metrics['total_iterations'] * metrics['mean_inference_time']:.1f}s")
        
        # Show cases with low consistency (high variance)
        print(f"\n📊 VARIANCE ANALYSIS:")
        high_variance_cases = [r for r in aggregated_results if r.prediction_consistency < 0.8]
        print(f"   Cases with <80% consistency: {len(high_variance_cases)}")
        
        if high_variance_cases:
            print("   Top inconsistent cases:")
            sorted_cases = sorted(high_variance_cases, key=lambda x: x.prediction_consistency)[:5]
            for case in sorted_cases:
                print(f"     \"{case.question[:60]}...\" - {case.prediction_consistency:.1%} consistency")
                predictions = [r.predicted for r in case.individual_results]
                pred_counts = Counter(predictions)
                print(f"       Predictions: {dict(pred_counts)}")
    
    def analyze_failures(self, results: List[SingleEvalResult] = None) -> None:
        """Analyze and report failure cases with focus on accuracy patterns"""
        if results is None:
            results = self.results
        
        failures = [r for r in results if not r.correct]
        
        print(f"\n📉 FAILURE ANALYSIS ({len(failures)} failures)")
        print("=" * 60)
        
        # Show all failures with reasoning
        print(f"\n❌ INCORRECT PREDICTIONS:")
        for failure in failures[:15]:  # Show top 15
            print(f"   \"{failure.question}\"")
            print(f"      Expected: {failure.intent_expected}, Got: {failure.predicted} (conf: {failure.confidence:.2f})")
            print(f"      Reasoning: {failure.reasoning}")
            print(f"      Category: {failure.category}, Difficulty: {failure.difficulty}")
            print()
        
        # Pattern analysis
        print(f"\n📊 FAILURE PATTERNS:")
        failure_categories = Counter(f.category for f in failures)
        for category, count in failure_categories.most_common():
            total_in_category = len([r for r in results if r.category == category])
            failure_rate = count / total_in_category if total_in_category > 0 else 0
            print(f"   {category}: {count}/{total_in_category} failures ({failure_rate:.1%})")
        
        failure_difficulties = Counter(f.difficulty for f in failures)
        for difficulty, count in failure_difficulties.most_common():
            total_in_difficulty = len([r for r in results if r.difficulty == difficulty])
            failure_rate = count / total_in_difficulty if total_in_difficulty > 0 else 0
            print(f"   {difficulty} difficulty: {count}/{total_in_difficulty} failures ({failure_rate:.1%})")
    
    def print_comprehensive_report(self, results: List[SingleEvalResult] = None) -> None:
        """Print a comprehensive evaluation report"""
        if results is None:
            results = self.results
        
        metrics = self.calculate_metrics(results)
        category_analysis = self.analyze_by_category(results)
        difficulty_analysis = self.analyze_by_difficulty(results)
        
        print(f"\n📊 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 50)
        
        # Overall metrics
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total_cases']})")
        
        # Confidence analysis
        conf_stats = metrics['confidence_stats']
        print(f"\n🎲 CONFIDENCE ANALYSIS:")
        print(f"   Mean Confidence: {conf_stats['mean_confidence']:.3f}")
        print(f"   Correct Predictions: {conf_stats['mean_correct_confidence']:.3f}")
        print(f"   Incorrect Predictions: {conf_stats['mean_incorrect_confidence']:.3f}")
        print(f"   Confidence Std Dev: {conf_stats['confidence_stdev']:.3f}")
        
        
        # Category analysis
        print(f"\n📂 PERFORMANCE BY CATEGORY:")
        for category, stats in sorted(category_analysis.items(), key=lambda x: x[1]['accuracy']):
            print(f"   {category:20s}: {stats['accuracy']:5.1%} ({stats['correct']:2d}/{stats['total']:2d})")
        
        # Difficulty analysis
        print(f"\n🎚️  PERFORMANCE BY DIFFICULTY:")
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in difficulty_analysis:
                stats = difficulty_analysis[difficulty]
                print(f"   {difficulty.capitalize():8s}: {stats['accuracy']:5.1%} ({stats['correct']:2d}/{stats['total']:2d})")