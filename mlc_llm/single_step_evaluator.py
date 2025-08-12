#!/usr/bin/env python3
"""
Single-Step Intent Detection Evaluator

Evaluation framework for single-step inference approach that combines:
1. Intent classification (action vs chat) 
2. Function selection (if action)

This evaluator compares the single-step approach against the two-step approach.
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

from model_wrapper import MLCModelWrapper
from testcase_loader import TestCaseLoader, TestCase

@dataclass
class SingleStepResult:
    """Result for a single single-step evaluation"""
    question: str
    intent_expected: str
    function_expected: str  # Expected function (derived from test case)
    category: str
    difficulty: str
    notes: str
    model_name: str
    
    # Single-step results
    intent_predicted: str  # "action" or "chat"
    function_predicted: Optional[str]  # Function name or None (for chat)
    confidence: float
    reasoning: str
    inference_time: float
    raw_response: str
    
    # Correctness evaluation
    intent_correct: bool
    function_correct: bool  # Only relevant for action cases
    overall_correct: bool  # Both intent and function correct

@dataclass
class SingleStepAggregatedResult:
    """Aggregated results across multiple iterations per test case"""
    question: str
    intent_expected: str
    function_expected: str
    category: str
    difficulty: str
    notes: str
    model_name: str
    iterations: int
    
    # Aggregated accuracy metrics
    intent_accuracy: float
    function_accuracy: float  # For action cases only
    overall_accuracy: float
    
    # Aggregated performance metrics
    mean_confidence: float
    mean_inference_time: float
    
    # Raw iteration data
    individual_results: List[SingleStepResult]

def load_single_step_prompt():
    """Load the single-step prompt from TypeScript file"""
    try:
        script_dir = Path(__file__).parent
        ts_file = script_dir.parent / "src" / "prompts" / "singleStep.ts"
        
        if not ts_file.exists():
            raise FileNotFoundError(f"Single-step prompt file not found: {ts_file}")
        
        content = ts_file.read_text()
        match = re.search(r'export const SINGLE_STEP_PROMPT = `(.*?)`;', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find SINGLE_STEP_PROMPT in TypeScript file")
        
        return match.group(1).strip()
        
    except Exception as e:
        print(f"❌ Failed to load single-step prompt: {e}")
        sys.exit(1)

class SingleStepEvaluator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "Phi-3.5-mini-instruct-q4f16_1-MLC"
        self.model_wrapper = MLCModelWrapper(self.model_name)
        
        # Load prompt
        self.single_step_prompt = load_single_step_prompt()
        
        self.total_eval_time = 0.0
    
    @property
    def model_size_mb(self) -> float:
        """Get model size from wrapper"""
        return self.model_wrapper.model_size_mb
    
    def _build_context_string(self, test_case: TestCase) -> str:
        """Build consistent context string (same as two-step evaluator)"""
        if test_case.context_pinned is True:
            # Use realistic thread data for testing
            realistic_threads = [
                {
                    "title": "Mexico to US Livestock Trade halted due to Screwworm spread",
                    "theme": "Screwworm Control Program Failure", 
                    "category": "INDUSTRY ANALYSIS",
                    "comments": 96,
                    "summary": "Analysis of agricultural trade disruption due to screwworm outbreak affecting Mexico-US livestock commerce. Discussion covers economic impacts on ranchers, effectiveness of sterile insect technique programs, and regulatory responses. Contributors share insights on biosecurity measures, historical precedents of screwworm eradication efforts, and potential timeline for trade resumption."
                },
                {
                    "title": "OpenSSH Post-Quantum Cryptography",
                    "theme": "Post-Quantum Crypto Adoption Overhead",
                    "category": "TECHNICAL DEEP DIVE", 
                    "comments": 28,
                    "summary": "Technical discussion about implementing post-quantum cryptography in OpenSSH. Thread examines performance implications, key size increases, and backward compatibility challenges. Engineers discuss NIST-approved algorithms, migration strategies, and real-world deployment considerations for quantum-resistant cryptographic systems."
                },
                {
                    "title": "Google paid a $250K reward for a bug",
                    "theme": "Bug bounty payouts comparison",
                    "category": "MARKET SENTIMENT",
                    "comments": 36,
                    "summary": "Comparison of bug bounty programs across major tech companies examining payout structures and vulnerability valuations. Discussion includes analysis of Google's $250K reward in context of industry standards, factors affecting bounty amounts, and experiences from security researchers participating in various programs."
                }
            ]
            
            # Select thread based on test case ID for consistency
            thread_index = hash(test_case.id) % len(realistic_threads)
            thread = realistic_threads[thread_index]
            
            return f'''Context: User has pinned this thread:
- Title: "{thread["title"]}"
- Theme: "{thread["theme"]}"
- Category: "{thread["category"]}"
- Comments: {thread["comments"]}
- Summary: "{thread["summary"]}"

'''
        elif test_case.context_pinned is False:
            return '''Context: No thread currently pinned.

'''
        else:
            return ''
    
    def _determine_expected_function(self, test_case: TestCase) -> str:
        """Determine expected function based on test case"""
        if test_case.intent_expected == "chat":
            return "no_action"
        elif test_case.category == "pinned_thread_summary":
            return "summarize_pinned_thread"
        else:
            # For other action cases, assume search
            return "get_thread_cards"
    
    def _build_single_step_prompt(self, test_case: TestCase) -> str:
        """Build single-step prompt"""
        context_info = self._build_context_string(test_case)
        return self.single_step_prompt.replace('{context}', context_info).replace('{message}', test_case.question)
    
    def _parse_single_step_response(self, response: str) -> dict:
        """Parse single-step response"""
        parsed = self.model_wrapper.parse_json_response(response)
        
        if "error" in parsed:
            return parsed
        
        # Validate intent field
        if "intent" not in parsed or parsed["intent"] not in ["action", "chat"]:
            return {"error": "Invalid intent field", "raw": response}
        
        # Validate confidence field
        if "confidence" not in parsed:
            return {"error": "Missing confidence field", "raw": response}
        
        try:
            confidence = float(parsed["confidence"])
            if not (0 <= confidence <= 1):
                return {"error": "Confidence must be between 0 and 1", "raw": response}
            parsed["confidence"] = confidence  # Ensure it's a float
        except (ValueError, TypeError):
            return {"error": "Invalid confidence field - must be a number", "raw": response}
        
        # For action intent, validate function field
        if parsed["intent"] == "action":
            if "function" not in parsed or not isinstance(parsed["function"], str):
                return {"error": "Missing or invalid function field for action intent", "raw": response}
        
        return parsed
    
    def evaluate_single_case(self, test_case: TestCase, temperature: float = 0.1) -> Optional[SingleStepResult]:
        """Evaluate a single test case using single-step approach"""
        function_expected = self._determine_expected_function(test_case)
        
        # Single-step inference
        prompt = self._build_single_step_prompt(test_case)
        response = self.model_wrapper.call_model(prompt, temperature, max_tokens=300)
        inference_time = getattr(self.model_wrapper, 'last_inference_time', 0.0)
        
        if not response:
            return None
        
        parsed = self._parse_single_step_response(response)
        if "error" in parsed:
            return None
        
        # Extract predictions
        intent_predicted = parsed.get('intent', 'chat')
        function_predicted = parsed.get('function') if intent_predicted == 'action' else None
        confidence = parsed.get('confidence', 0.0)
        reasoning = parsed.get('reasoning', '')
        
        # Evaluate correctness
        intent_correct = intent_predicted == test_case.intent_expected
        
        # Function correctness only matters for action cases
        if test_case.intent_expected == "action":
            function_correct = function_predicted == function_expected
        else:
            # For chat cases, function should be None/not specified
            function_correct = function_predicted is None
        
        overall_correct = intent_correct and function_correct
        
        return SingleStepResult(
            question=test_case.question,
            intent_expected=test_case.intent_expected,
            function_expected=function_expected,
            category=test_case.category,
            difficulty=test_case.difficulty,
            notes=test_case.notes,
            model_name=self.model_name,
            intent_predicted=intent_predicted,
            function_predicted=function_predicted,
            confidence=confidence,
            reasoning=reasoning,
            inference_time=inference_time,
            raw_response=response,
            intent_correct=intent_correct,
            function_correct=function_correct,
            overall_correct=overall_correct
        )
    
    def run_evaluation(self, iterations: int = 5, temperature: float = 0.1, dataset_filter: str = None, verbose: bool = False) -> List[SingleStepAggregatedResult]:
        """Run single-step evaluation with multiple iterations per test case"""
        start_time = time.time()
        
        # Load test cases
        loader = TestCaseLoader("mlc_llm/bfcl_testcases.json")
        if dataset_filter:
            test_cases = loader.get_cases_by_category(dataset_filter)
            print(f"📋 Loaded {len(test_cases)} test cases for category: {dataset_filter}")
        else:
            test_cases = loader.test_cases
            print(f"📋 Loaded {len(test_cases)} test cases total")
        
        if not test_cases:
            print("❌ No test cases found")
            return []
        
        aggregated_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            if verbose:
                print(f"\\n🔍 [{i}/{len(test_cases)}] Evaluating: {test_case.question}")
                print(f"   Expected: {test_case.intent_expected}")
                if test_case.context_pinned is not None:
                    print(f"   Context: {'Pinned thread' if test_case.context_pinned else 'No pinned thread'}")
            
            # Run multiple iterations for this test case
            individual_results = []
            for iteration in range(iterations):
                result = self.evaluate_single_case(test_case, temperature)
                if result:
                    individual_results.append(result)
                
                if verbose and iteration == 0:  # Show first result for debugging
                    if result:
                        print(f"   Predicted: {result.intent_predicted}, Function: {result.function_predicted}")
                        print(f"   Correct: Intent={result.intent_correct}, Function={result.function_correct}, Overall={result.overall_correct}")
            
            if individual_results:
                # Calculate aggregated metrics
                intent_correct_count = sum(1 for r in individual_results if r.intent_correct)
                function_correct_count = sum(1 for r in individual_results if r.function_correct)
                overall_correct_count = sum(1 for r in individual_results if r.overall_correct)
                
                aggregated_result = SingleStepAggregatedResult(
                    question=test_case.question,
                    intent_expected=test_case.intent_expected,
                    function_expected=self._determine_expected_function(test_case),
                    category=test_case.category,
                    difficulty=test_case.difficulty,
                    notes=test_case.notes,
                    model_name=self.model_name,
                    iterations=len(individual_results),
                    intent_accuracy=intent_correct_count / len(individual_results),
                    function_accuracy=function_correct_count / len(individual_results),
                    overall_accuracy=overall_correct_count / len(individual_results),
                    mean_confidence=statistics.mean([r.confidence for r in individual_results]),
                    mean_inference_time=statistics.mean([r.inference_time for r in individual_results]),
                    individual_results=individual_results
                )
                
                aggregated_results.append(aggregated_result)
        
        self.total_eval_time = time.time() - start_time
        return aggregated_results
    
    def calculate_aggregated_metrics(self, results: List[SingleStepAggregatedResult]) -> Dict[str, float]:
        """Calculate overall metrics across all test cases"""
        if not results:
            return {}
        
        # Overall accuracy metrics
        intent_accuracies = [r.intent_accuracy for r in results]
        function_accuracies = [r.function_accuracy for r in results]
        overall_accuracies = [r.overall_accuracy for r in results]
        
        # Performance metrics
        inference_times = [r.mean_inference_time for r in results]
        confidences = [r.mean_confidence for r in results]
        
        return {
            'intent_overall_accuracy': statistics.mean(intent_accuracies),
            'function_overall_accuracy': statistics.mean(function_accuracies),
            'overall_accuracy': statistics.mean(overall_accuracies),
            'mean_inference_time': statistics.mean(inference_times),
            'mean_confidence': statistics.mean(confidences),
            'total_test_cases': len(results),
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.model_wrapper, 'cleanup'):
            self.model_wrapper.cleanup()