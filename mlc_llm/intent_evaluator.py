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

from tqdm import tqdm
from model_wrapper import MLCModelWrapper
from testcase_loader import TestCaseLoader, TestCase

@dataclass
class EvalResult:
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
    model_name: str = ""

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
        self.results: List[EvalResult] = []
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
    
    def evaluate_test_case(self, test_case: TestCase, temperature: float = 0.1, verbose: bool = True) -> Optional[EvalResult]:
        """Evaluate a single test case"""
        if verbose:
            print(f"\n🧪 Testing: \"{test_case.question}\" ({test_case.category}, {test_case.difficulty})")
        
        # Build prompt
        prompt = self.prompt_template.replace('{message}', test_case.question)
        
        # Call model
        response = self.model_wrapper.call_model(prompt, temperature, max_tokens=500)
        if not response:
            return None
        
        # Parse response
        parsed = self.parse_response(response)
        
        if "error" in parsed:
            if verbose:
                print(f"   ❌ Parse Error: {parsed['error']}")
            return None
        
        # Create result
        predicted = parsed.get('intentCategory', 'ERROR')
        correct = predicted == test_case.intent_expected
        
        result = EvalResult(
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
            model_name=self.model_name
        )
        
        if verbose:
            status = "✅ CORRECT" if correct else "❌ WRONG"
            print(f"   Predicted: {predicted} (confidence: {result.confidence:.2f}) → {status}")
            if not correct:
                print(f"   Expected: {test_case.intent_expected}")
                print(f"   Reasoning: {result.reasoning}")
        
        return result
    
    def run_full_evaluation(self, temperature: float = 0.1, dataset_filter: str = None, verbose: bool = True) -> List[EvalResult]:
        """Run evaluation on all or filtered test cases"""
        loader = TestCaseLoader("mlc_llm/bfcl_testcases.json")
        
        if dataset_filter:
            test_cases = loader.get_cases_by_category(dataset_filter)
            print(f"📊 Running evaluation on {len(test_cases)} test cases (filter: {dataset_filter})")
        else:
            test_cases = loader.test_cases
            print(f"📊 Running full evaluation on {len(test_cases)} test cases")
        
        results = []
        
        # Use tqdm for progress tracking
        if not verbose:
            iterator = tqdm(test_cases, desc=f"Evaluating {self.model_name}", unit="test")
        else:
            iterator = test_cases
        
        for i, test_case in enumerate(iterator, 1):
            if verbose:
                print(f"\nProgress: {i}/{len(test_cases)}")
            
            result = self.evaluate_test_case(test_case, temperature, verbose)
            if result:
                results.append(result)
        
        self.results = results
        return results
    
    def calculate_metrics(self, results: List[EvalResult] = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        if results is None:
            results = self.results
        
        if not results:
            return {}
        
        # Basic metrics
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        # Confusion matrix
        true_positive = sum(1 for r in results if r.intent_expected == "action" and r.predicted == "action")
        false_positive = sum(1 for r in results if r.intent_expected == "chat" and r.predicted == "action")
        true_negative = sum(1 for r in results if r.intent_expected == "chat" and r.predicted == "chat")
        false_negative = sum(1 for r in results if r.intent_expected == "action" and r.predicted == "chat")
        
        # Precision, Recall, F1 for "action" class
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Confidence analysis
        confidences = [r.confidence for r in results]
        correct_confidences = [r.confidence for r in results if r.correct]
        incorrect_confidences = [r.confidence for r in results if not r.correct]
        
        return {
            "accuracy": accuracy,
            "total_cases": total,
            "correct": correct,
            "confusion_matrix": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confidence_stats": {
                "mean_confidence": statistics.mean(confidences) if confidences else 0,
                "mean_correct_confidence": statistics.mean(correct_confidences) if correct_confidences else 0,
                "mean_incorrect_confidence": statistics.mean(incorrect_confidences) if incorrect_confidences else 0,
                "confidence_stdev": statistics.stdev(confidences) if len(confidences) > 1 else 0
            }
        }
    
    def analyze_by_category(self, results: List[EvalResult] = None) -> Dict:
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
    
    def analyze_by_difficulty(self, results: List[EvalResult] = None) -> Dict:
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
    
    def analyze_failures(self, results: List[EvalResult] = None) -> None:
        """Analyze and report failure cases"""
        if results is None:
            results = self.results
        
        failures = [r for r in results if not r.correct]
        
        print(f"\n📉 FAILURE ANALYSIS ({len(failures)} failures)")
        print("=" * 60)
        
        # Group failures by type
        false_positives = [r for r in failures if r.intent_expected == "chat" and r.predicted == "action"]
        false_negatives = [r for r in failures if r.intent_expected == "action" and r.predicted == "chat"]
        
        print(f"\n❌ FALSE POSITIVES (classified as action, should be chat): {len(false_positives)}")
        for fp in false_positives[:10]:  # Show top 10
            print(f"   \"{fp.question}\" → {fp.predicted} (conf: {fp.confidence:.2f})")
            print(f"      Reasoning: {fp.reasoning}")
            print(f"      Category: {fp.category}, Notes: {fp.notes}")
            print()
        
        print(f"\n❌ FALSE NEGATIVES (classified as chat, should be action): {len(false_negatives)}")
        for fn in false_negatives[:10]:  # Show top 10
            print(f"   \"{fn.question}\" → {fn.predicted} (conf: {fn.confidence:.2f})")
            print(f"      Reasoning: {fn.reasoning}")
            print(f"      Category: {fn.category}, Notes: {fn.notes}")
            print()
        
        # Pattern analysis
        print(f"\n📊 FAILURE PATTERNS:")
        failure_categories = Counter(f.category for f in failures)
        for category, count in failure_categories.most_common():
            print(f"   {category}: {count} failures")
        
        failure_difficulties = Counter(f.difficulty for f in failures)
        for difficulty, count in failure_difficulties.most_common():
            print(f"   {difficulty} difficulty: {count} failures")
    
    def print_comprehensive_report(self, results: List[EvalResult] = None) -> None:
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
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall: {metrics['recall']:.1%}")
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        
        # Confidence analysis
        conf_stats = metrics['confidence_stats']
        print(f"\n🎲 CONFIDENCE ANALYSIS:")
        print(f"   Mean Confidence: {conf_stats['mean_confidence']:.3f}")
        print(f"   Correct Predictions: {conf_stats['mean_correct_confidence']:.3f}")
        print(f"   Incorrect Predictions: {conf_stats['mean_incorrect_confidence']:.3f}")
        print(f"   Confidence Std Dev: {conf_stats['confidence_stdev']:.3f}")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"                    Predicted")
        print(f"                Action    Chat")
        print(f"   Actual Action    {cm['true_positive']:2d}      {cm['false_negative']:2d}")
        print(f"          Chat      {cm['false_positive']:2d}      {cm['true_negative']:2d}")
        
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