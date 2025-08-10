#!/usr/bin/env python3
"""
Test Case Loader Module

Pluggable module for loading and managing test cases in BFCL (Berkeley Function Calling Leaderboard) format.
Supports both legacy format and new unified format for different evaluation phases.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class TestCase:
    """Unified test case format supporting all evaluation phases"""
    id: str
    question: str
    category: str
    difficulty: str
    notes: str = ""
    
    # Phase 1: Intent Classification
    intent_expected: Optional[str] = None
    
    # Phase 2: Action Type Classification  
    action_type_expected: Optional[str] = None
    
    # Phase 3: Function Calling (BFCL format)
    function: Optional[List[Dict[str, Any]]] = None
    expected_function_call: Optional[str] = None
    valid_alternatives: Optional[List[str]] = None
    
    # Evaluation metadata
    evaluation_phases: Optional[List[str]] = None

class TestCaseLoader:
    """Load and manage test cases for different evaluation phases"""
    
    def __init__(self, testcase_file: str = "mlc_llm/bfcl_testcases.json"):
        self.testcase_file = testcase_file
        self.metadata = {}
        self.functions = {}
        self.test_cases = []
        self._load_testcases()
    
    def _load_testcases(self):
        """Load test cases from JSON file"""
        try:
            with open(self.testcase_file, 'r') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', {})
            self.functions = data.get('functions', {})
            test_cases_data = data.get('test_cases', [])
            
            self.test_cases = [
                TestCase(
                    id=case['id'],
                    question=case['question'],
                    category=case['category'],
                    difficulty=case['difficulty'],
                    notes=case.get('notes', ''),
                    intent_expected=case.get('intent_expected'),
                    action_type_expected=case.get('action_type_expected'),
                    function=case.get('function'),
                    expected_function_call=case.get('expected_function_call'),
                    valid_alternatives=case.get('valid_alternatives'),
                    evaluation_phases=case.get('evaluation_phases', [])
                )
                for case in test_cases_data
            ]
            
            print(f"✅ Loaded {len(self.test_cases)} test cases from {self.testcase_file}")
            print(f"📋 Metadata: {self.metadata.get('version', 'unknown version')}")
            
        except FileNotFoundError:
            print(f"❌ Test case file not found: {self.testcase_file}")
            print("   Please ensure the JSON file exists.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in test case file: {e}")
            sys.exit(1)
    
    def get_cases_for_phase(self, phase: str) -> List[TestCase]:
        """Get test cases for a specific evaluation phase
        
        Phase 1: intent_classification - chat vs action (all test cases)
        Phase 2: function_calling - function selection + parameters (action cases only)
        """
        phase_mapping = {
            'intent': 'intent_classification',
            'intent_classification': 'intent_classification', 
            'phase1': 'intent_classification',
            'function_calling': 'function_calling',
            'function': 'function_calling',
            'phase2': 'function_calling'
        }
        
        normalized_phase = phase_mapping.get(phase, phase)
        
        return [
            case for case in self.test_cases 
            if normalized_phase in (case.evaluation_phases or [])
        ]
    
    def get_cases_by_category(self, category: str) -> List[TestCase]:
        """Get test cases by category"""
        return [case for case in self.test_cases if case.category == category]
    
    def get_cases_by_difficulty(self, difficulty: str) -> List[TestCase]:
        """Get test cases by difficulty level"""
        return [case for case in self.test_cases if case.difficulty == difficulty]
    
    def get_cases_by_intent(self, intent: str) -> List[TestCase]:
        """Get test cases by expected intent"""
        return [case for case in self.test_cases if case.intent_expected == intent]
    
    def filter_cases(self, 
                    phase: Optional[str] = None,
                    category: Optional[str] = None,
                    difficulty: Optional[str] = None,
                    intent: Optional[str] = None) -> List[TestCase]:
        """Filter test cases by multiple criteria"""
        
        cases = self.test_cases
        
        if phase:
            cases = [case for case in cases if case in self.get_cases_for_phase(phase)]
        
        if category:
            cases = [case for case in cases if case.category == category]
        
        if difficulty:
            cases = [case for case in cases if case.difficulty == difficulty]
            
        if intent:
            cases = [case for case in cases if case.intent_expected == intent]
        
        return cases
    
    def get_function_schema(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get function schema by name"""
        return self.functions.get(function_name)
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories"""
        return list(set(case.category for case in self.test_cases))
    
    def get_all_difficulties(self) -> List[str]:
        """Get all unique difficulty levels"""
        return list(set(case.difficulty for case in self.test_cases))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'total_cases': len(self.test_cases),
            'categories': {cat: len(self.get_cases_by_category(cat)) for cat in self.get_all_categories()},
            'difficulties': {diff: len(self.get_cases_by_difficulty(diff)) for diff in self.get_all_difficulties()},
            'intents': {
                'chat': len(self.get_cases_by_intent('chat')),
                'action': len(self.get_cases_by_intent('action'))
            },
            'phases': {
                'intent_classification': len(self.get_cases_for_phase('intent')),
                'action_type_classification': len(self.get_cases_for_phase('action_type')),
                'function_calling': len(self.get_cases_for_phase('function_calling'))
            }
        }
    
    def validate_testcases(self) -> List[str]:
        """Validate test cases and return list of issues"""
        issues = []
        
        for case in self.test_cases:
            # Check required fields
            if not case.id:
                issues.append(f"Missing ID in test case")
            if not case.question:
                issues.append(f"Missing question in test case {case.id}")
            
            # Check intent classification phase
            if 'intent_classification' in (case.evaluation_phases or []):
                if not case.intent_expected:
                    issues.append(f"Missing intent_expected in case {case.id}")
                elif case.intent_expected not in ['chat', 'action']:
                    issues.append(f"Invalid intent_expected '{case.intent_expected}' in case {case.id}")
            
            # Check function calling phase
            if 'function_calling' in (case.evaluation_phases or []):
                if not case.function:
                    issues.append(f"Missing function schema in case {case.id}")
                if not case.expected_function_call:
                    issues.append(f"Missing expected_function_call in case {case.id}")
        
        return issues

# Legacy compatibility functions
def load_test_cases_from_json(json_file: str = "mlc_llm/bfcl_testcases.json") -> List[TestCase]:
    """Legacy function for backward compatibility"""
    loader = TestCaseLoader(json_file)
    return loader.test_cases

def convert_legacy_testcase(legacy_case: Dict[str, Any]) -> TestCase:
    """Convert legacy test case format to new unified format"""
    return TestCase(
        id=legacy_case.get('id', f"legacy_{hash(legacy_case['query'])}"),
        question=legacy_case['query'],
        category=legacy_case['category'],
        difficulty=legacy_case['difficulty'],
        notes=legacy_case.get('notes', ''),
        intent_expected=legacy_case['expected'],  # 'action' or 'chat'
        evaluation_phases=['intent_classification']
    )