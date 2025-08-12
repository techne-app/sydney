#!/usr/bin/env python3
"""
Two-Step Intent Detection Evaluator

Dedicated evaluation framework for two-step inference approach:
1. Step 1: Intent classification (action vs chat)
2. Step 2: Function selection (for action cases only)

This evaluator measures accuracy for both steps separately and combined.
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
class TwoStepResult:
    """Result for a single two-step evaluation"""
    question: str
    intent_expected: str
    function_expected: str  # Expected function (derived from test case)
    category: str
    difficulty: str
    notes: str
    model_name: str
    
    # Step 1 results
    step1_predicted: str  # "action" or "chat"
    step1_confidence: float
    step1_reasoning: str
    step1_correct: bool
    step1_time: float
    step1_raw: str
    
    # Step 2 results (only if step1 was action)
    step2_predicted: Optional[str]  # Function name or None
    step2_confidence: Optional[float]
    step2_reasoning: Optional[str]
    step2_correct: Optional[bool]  # None if step2 not executed
    step2_time: float
    step2_raw: Optional[str]
    
    # Combined results
    overall_correct: bool  # Both steps correct
    total_time: float
    final_confidence: float  # Min of step1 and step2 (if both executed)

@dataclass
class TwoStepAggregatedResult:
    """Aggregated results across multiple iterations per test case"""
    question: str
    intent_expected: str
    function_expected: str
    category: str
    difficulty: str
    notes: str
    model_name: str
    iterations: int
    
    # Step 1 aggregated metrics
    step1_accuracy: float
    step1_mean_confidence: float
    step1_mean_time: float
    
    # Step 2 aggregated metrics (only for cases where step1 was action)
    step2_accuracy: Optional[float]  # None if no action cases
    step2_mean_confidence: Optional[float]
    step2_mean_time: Optional[float]
    step2_cases_count: int  # Number of iterations that reached step 2
    
    # Overall metrics
    overall_accuracy: float  # Both steps correct
    mean_total_time: float
    mean_final_confidence: float
    
    # Raw iteration data
    individual_results: List[TwoStepResult]

def load_intent_only_prompt():
    """Load the intent-only prompt from TypeScript file"""
    try:
        script_dir = Path(__file__).parent
        ts_file = script_dir.parent / "src" / "prompts" / "intentOnly.ts"
        
        if not ts_file.exists():
            raise FileNotFoundError(f"Intent-only prompt file not found: {ts_file}")
        
        content = ts_file.read_text()
        match = re.search(r'export const INTENT_ONLY_PROMPT = `(.*?)`;', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find INTENT_ONLY_PROMPT in TypeScript file")
        
        return match.group(1).strip()
        
    except Exception as e:
        print(f"❌ Failed to load intent-only prompt: {e}")
        sys.exit(1)

def load_action_only_prompt():
    """Load the action-only prompt from TypeScript file"""
    try:
        script_dir = Path(__file__).parent
        ts_file = script_dir.parent / "src" / "prompts" / "actionOnly.ts"
        
        if not ts_file.exists():
            raise FileNotFoundError(f"Action-only prompt file not found: {ts_file}")
        
        content = ts_file.read_text()
        match = re.search(r'export const ACTION_ONLY_PROMPT = `(.*?)`;', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find ACTION_ONLY_PROMPT in TypeScript file")
        
        return match.group(1).strip()
        
    except Exception as e:
        print(f"❌ Failed to load action-only prompt: {e}")
        sys.exit(1)

class TwoStepEvaluator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "Phi-3.5-mini-instruct-q4f16_1-MLC"
        self.model_wrapper = MLCModelWrapper(self.model_name)
        
        # Load prompts
        self.intent_only_prompt = load_intent_only_prompt()
        self.action_only_prompt = load_action_only_prompt()
        
        self.total_eval_time = 0.0
    
    @property
    def model_size_mb(self) -> float:
        """Get model size from wrapper"""
        return self.model_wrapper.model_size_mb
    
    def _build_context_string(self, test_case: TestCase) -> str:
        """Build consistent context string"""
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
    
    def _build_intent_prompt(self, test_case: TestCase) -> str:
        """Build step 1 intent detection prompt"""
        context_info = self._build_context_string(test_case)
        return self.intent_only_prompt.replace('{context}', context_info).replace('{message}', test_case.question)
    
    def _build_action_prompt(self, test_case: TestCase) -> str:
        """Build step 2 action selection prompt"""
        context_info = self._build_context_string(test_case)
        return self.action_only_prompt.replace('{context}', context_info).replace('{message}', test_case.question)
    
    def _parse_intent_response(self, response: str) -> dict:
        """Parse step 1 intent response"""
        parsed = self.model_wrapper.parse_json_response(response)
        
        if "error" in parsed:
            return parsed
        
        if "intent" not in parsed or parsed["intent"] not in ["action", "chat"]:
            return {"error": "Invalid intent field", "raw": response}
        
        if "confidence" not in parsed or not (0 <= parsed["confidence"] <= 1):
            return {"error": "Invalid confidence field", "raw": response}
        
        return parsed
    
    def _parse_action_response(self, response: str) -> dict:
        """Parse step 2 action response"""
        parsed = self.model_wrapper.parse_json_response(response)
        
        if "error" in parsed:
            return parsed
        
        if "function" not in parsed or not isinstance(parsed["function"], str):
            return {"error": "Invalid function field", "raw": response}
        
        if "confidence" not in parsed or not (0 <= parsed["confidence"] <= 1):
            return {"error": "Invalid confidence field", "raw": response}
        
        return parsed
    
    def evaluate_single_case(self, test_case: TestCase, temperature: float = 0.1) -> Optional[TwoStepResult]:
        """Evaluate a single test case using two-step approach"""
        function_expected = self._determine_expected_function(test_case)
        
        # Step 1: Intent detection
        intent_prompt = self._build_intent_prompt(test_case)
        intent_response = self.model_wrapper.call_model(intent_prompt, temperature, max_tokens=200)
        step1_time = getattr(self.model_wrapper, 'last_inference_time', 0.0)
        
        if not intent_response:
            return None
        
        intent_parsed = self._parse_intent_response(intent_response)
        if "error" in intent_parsed:
            return None
        
        step1_predicted = intent_parsed.get('intent', 'chat')
        step1_confidence = intent_parsed.get('confidence', 0.0)
        step1_reasoning = intent_parsed.get('reasoning', '')
        step1_correct = step1_predicted == test_case.intent_expected
        
        # Step 2: Function selection (only if step1 predicted action)
        step2_predicted = None
        step2_confidence = None
        step2_reasoning = None
        step2_correct = None
        step2_time = 0.0
        step2_raw = None
        
        if step1_predicted == 'action':
            action_prompt = self._build_action_prompt(test_case)
            action_response = self.model_wrapper.call_model(action_prompt, temperature, max_tokens=200)
            step2_time = getattr(self.model_wrapper, 'last_inference_time', 0.0)
            
            if action_response:
                action_parsed = self._parse_action_response(action_response)
                if "error" not in action_parsed:
                    step2_predicted = action_parsed.get('function', 'get_thread_cards')
                    step2_confidence = action_parsed.get('confidence', 0.0)
                    step2_reasoning = action_parsed.get('reasoning', '')
                    step2_correct = step2_predicted == function_expected
                    step2_raw = action_response
        
        # Calculate combined metrics
        overall_correct = step1_correct and (step2_correct if step2_correct is not None else True)
        total_time = step1_time + step2_time
        
        # Final confidence is minimum of both steps (if both executed)
        if step2_confidence is not None:
            final_confidence = min(step1_confidence, step2_confidence)
        else:
            final_confidence = step1_confidence
        
        return TwoStepResult(
            question=test_case.question,
            intent_expected=test_case.intent_expected,
            function_expected=function_expected,
            category=test_case.category,
            difficulty=test_case.difficulty,
            notes=test_case.notes,
            model_name=self.model_name,
            
            step1_predicted=step1_predicted,
            step1_confidence=step1_confidence,
            step1_reasoning=step1_reasoning,
            step1_correct=step1_correct,
            step1_time=step1_time,
            step1_raw=intent_response,
            
            step2_predicted=step2_predicted,
            step2_confidence=step2_confidence,
            step2_reasoning=step2_reasoning,
            step2_correct=step2_correct,
            step2_time=step2_time,
            step2_raw=step2_raw,
            
            overall_correct=overall_correct,
            total_time=total_time,
            final_confidence=final_confidence
        )
    
    def evaluate_test_case(self, test_case: TestCase, iterations: int = 10, temperature: float = 0.1, verbose: bool = True) -> Optional[TwoStepAggregatedResult]:
        """Evaluate a single test case multiple times and aggregate results"""
        if verbose:
            print(f"\n🧪 Two-Step Testing: \"{test_case.question}\" ({test_case.category}, {test_case.difficulty}) - {iterations} iterations")
        
        individual_results = []
        
        # Run multiple iterations
        for iteration in range(iterations):
            if verbose:
                print(f"   Iteration {iteration + 1}/{iterations}", end="... ")
            
            result = self.evaluate_single_case(test_case, temperature)
            if result:
                individual_results.append(result)
            
            if verbose:
                if result:
                    step1_status = "✅" if result.step1_correct else "❌"
                    step2_status = "✅" if result.step2_correct else "❌" if result.step2_correct is not None else "➖"
                    overall_status = "✅" if result.overall_correct else "❌"
                    print(f"S1:{step1_status}({result.step1_predicted}) S2:{step2_status}({result.step2_predicted or 'N/A'}) Overall:{overall_status}")
                else:
                    print("ERROR")
        
        if not individual_results:
            return None
        
        # Calculate Step 1 aggregated metrics
        step1_correct_count = sum(1 for r in individual_results if r.step1_correct)
        step1_accuracy = step1_correct_count / len(individual_results)
        step1_mean_confidence = statistics.mean([r.step1_confidence for r in individual_results])
        step1_mean_time = statistics.mean([r.step1_time for r in individual_results])
        
        # Calculate Step 2 aggregated metrics (only for cases that reached step 2)
        step2_results = [r for r in individual_results if r.step2_correct is not None]
        step2_cases_count = len(step2_results)
        
        if step2_results:
            step2_correct_count = sum(1 for r in step2_results if r.step2_correct)
            step2_accuracy = step2_correct_count / len(step2_results)
            step2_mean_confidence = statistics.mean([r.step2_confidence for r in step2_results])
            step2_mean_time = statistics.mean([r.step2_time for r in step2_results])
        else:
            step2_accuracy = None
            step2_mean_confidence = None
            step2_mean_time = None
        
        # Calculate overall metrics
        overall_correct_count = sum(1 for r in individual_results if r.overall_correct)
        overall_accuracy = overall_correct_count / len(individual_results)
        mean_total_time = statistics.mean([r.total_time for r in individual_results])
        mean_final_confidence = statistics.mean([r.final_confidence for r in individual_results])
        
        aggregated_result = TwoStepAggregatedResult(
            question=test_case.question,
            intent_expected=test_case.intent_expected,
            function_expected=individual_results[0].function_expected,
            category=test_case.category,
            difficulty=test_case.difficulty,
            notes=test_case.notes,
            model_name=self.model_name,
            iterations=len(individual_results),
            
            step1_accuracy=step1_accuracy,
            step1_mean_confidence=step1_mean_confidence,
            step1_mean_time=step1_mean_time,
            
            step2_accuracy=step2_accuracy,
            step2_mean_confidence=step2_mean_confidence,
            step2_mean_time=step2_mean_time,
            step2_cases_count=step2_cases_count,
            
            overall_accuracy=overall_accuracy,
            mean_total_time=mean_total_time,
            mean_final_confidence=mean_final_confidence,
            
            individual_results=individual_results
        )
        
        if verbose:
            step2_display = f"{step2_accuracy:.1%}" if step2_accuracy is not None else "N/A"
            print(f"   📊 Results: Step1:{step1_accuracy:.1%} Step2:{step2_display} Overall:{overall_accuracy:.1%}")
        
        return aggregated_result
    
    def run_evaluation(self, iterations: int = 10, temperature: float = 0.1, dataset_filter: str = None, verbose: bool = True) -> List[TwoStepAggregatedResult]:
        """Run two-step evaluation on test cases"""
        loader = TestCaseLoader("mlc_llm/bfcl_testcases.json")
        
        if dataset_filter:
            test_cases = loader.get_cases_by_category(dataset_filter)
            print(f"📊 Running TWO-STEP evaluation on {len(test_cases)} test cases (filter: {dataset_filter})")
        else:
            test_cases = loader.test_cases
            print(f"📊 Running TWO-STEP evaluation on {len(test_cases)} test cases")
        
        print(f"🚀 Loading {self.model_name}...")
        print(f"🔄 Each test case will be run {iterations} times for statistical analysis")
        print(f"📋 Step 1: Intent Detection (action vs chat)")
        print(f"📋 Step 2: Function Selection (when intent=action)")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            if verbose:
                print(f"\\nProgress: {i}/{len(test_cases)}")
            
            result = self.evaluate_test_case(test_case, iterations, temperature, verbose)
            if result:
                results.append(result)
        
        return results
    
    def calculate_aggregated_metrics(self, results: List[TwoStepAggregatedResult]) -> Dict:
        """Calculate overall metrics across all test cases"""
        if not results:
            return {}
        
        # Overall accuracies
        step1_accuracies = [r.step1_accuracy for r in results]
        step2_accuracies = [r.step2_accuracy for r in results if r.step2_accuracy is not None]
        overall_accuracies = [r.overall_accuracy for r in results]
        
        return {
            "step1_overall_accuracy": statistics.mean(step1_accuracies),
            "step2_overall_accuracy": statistics.mean(step2_accuracies) if step2_accuracies else None,
            "overall_accuracy": statistics.mean(overall_accuracies),
            "step1_std": statistics.stdev(step1_accuracies) if len(step1_accuracies) > 1 else 0.0,
            "step2_std": statistics.stdev(step2_accuracies) if len(step2_accuracies) > 1 else 0.0,
            "overall_std": statistics.stdev(overall_accuracies) if len(overall_accuracies) > 1 else 0.0,
            "total_test_cases": len(results),
            "step2_applicable_cases": len(step2_accuracies),
            "mean_inference_time": statistics.mean([r.mean_total_time for r in results]),
            "total_iterations": sum(r.iterations for r in results)
        }
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model_wrapper:
            self.model_wrapper.cleanup()