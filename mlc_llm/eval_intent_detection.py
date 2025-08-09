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
import re
import sys
import csv
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import statistics

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress function
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from mlc_llm import MLCEngine
    MLC_AVAILABLE = True
except ImportError:
    print("⚠️  MLC LLM not installed. Install with:")
    print("   uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly")
    print("   uv pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly")
    MLC_AVAILABLE = False

@dataclass
class TestCase:
    query: str
    expected: str  # 'action' or 'chat'
    category: str  # test category for analysis
    difficulty: str  # 'easy', 'medium', 'hard'
    notes: str = ""

@dataclass
class EvalResult:
    query: str
    expected: str
    predicted: str
    confidence: float
    reasoning: str
    raw_response: str
    correct: bool
    category: str
    difficulty: str
    notes: str
    model_name: str = ""

@dataclass
class ModelComparison:
    model_a: str
    model_b: str
    results_a: List[EvalResult]
    results_b: List[EvalResult]
    accuracy_a: float
    accuracy_b: float
    disagreements: List[Tuple[str, EvalResult, EvalResult]]

# Comprehensive test dataset
COMPREHENSIVE_TEST_CASES = [
    # === CLEAR SEARCH INTENTS (should be ACTION) ===
    TestCase("find discussions about AI", "action", "explicit_search", "easy", "Classic search command"),
    TestCase("search for startups", "action", "explicit_search", "easy", "Direct search verb"),
    TestCase("look up JavaScript discussions", "action", "explicit_search", "easy", "Look up variant"),
    TestCase("show me posts about machine learning", "action", "explicit_search", "easy", "Show me variant"),
    TestCase("get me some blockchain threads", "action", "explicit_search", "easy", "Get me variant"),
    TestCase("can you find me discussions about React?", "action", "polite_search", "easy", "Polite search request"),
    TestCase("I'm looking for threads on cryptocurrency", "action", "indirect_search", "medium", "Indirect search phrasing"),
    TestCase("any posts about remote work?", "action", "question_search", "medium", "Question form search"),
    TestCase("what's been said about Python lately?", "action", "question_search", "medium", "Recent discussion search"),
    TestCase("where can I find info on web3?", "action", "location_search", "medium", "Where-based search"),
    
    # === CLEAR CONVERSATIONAL INTENTS (should be CHAT) ===
    TestCase("hello how are you", "chat", "greeting", "easy", "Simple greeting"),
    TestCase("what's your opinion on React?", "chat", "opinion_request", "medium", "Asking for opinion"),
    TestCase("can you explain machine learning?", "chat", "explanation_request", "medium", "Asking for explanation"),
    TestCase("tell me about startups", "chat", "information_request", "medium", "General info request"),
    TestCase("how does JavaScript work?", "chat", "how_question", "medium", "How question"),
    TestCase("what is blockchain?", "chat", "definition_request", "medium", "Definition question"),
    TestCase("thanks for your help", "chat", "gratitude", "easy", "Thank you message"),
    TestCase("I don't understand", "chat", "confusion", "easy", "Confusion statement"),
    TestCase("that's interesting", "chat", "reaction", "easy", "Reaction to information"),
    TestCase("good morning", "chat", "greeting", "easy", "Time-based greeting"),
    
    # === AMBIGUOUS CASES (harder to classify) ===
    TestCase("what about React?", "action", "ambiguous", "hard", "Could be opinion or search - context dependent"),
    TestCase("thoughts on AI?", "chat", "ambiguous", "hard", "Asking for thoughts/opinions"),
    TestCase("anything on startups?", "action", "ambiguous", "hard", "Implicit search request"),
    TestCase("React?", "action", "ambiguous", "hard", "Single word - likely search"),
    TestCase("tell me what you think about Python", "chat", "ambiguous", "hard", "Opinion request with 'tell me'"),
    TestCase("what's new with cryptocurrency", "action", "ambiguous", "medium", "Could be news search or general question"),
    TestCase("help me understand React", "chat", "ambiguous", "medium", "Help request - explanation vs search"),
    TestCase("I want to know about machine learning", "chat", "ambiguous", "medium", "Want to know - learning intent"),
    TestCase("what's going on with tech layoffs", "action", "ambiguous", "medium", "Current events - likely search"),
    TestCase("give me your take on remote work", "chat", "ambiguous", "medium", "Asking for opinion/perspective"),
    
    # === EDGE CASES ===
    TestCase("", "chat", "edge_case", "hard", "Empty query"),
    TestCase("???", "chat", "edge_case", "hard", "Just punctuation"),
    TestCase("find", "action", "edge_case", "hard", "Incomplete search command"),
    TestCase("search", "action", "edge_case", "hard", "Search without topic"),
    TestCase("what", "chat", "edge_case", "hard", "Incomplete question"),
    TestCase("seach for AI", "action", "edge_case", "medium", "Typo in search command"),
    TestCase("finde discussions about React", "action", "edge_case", "medium", "Typo in search verb"),
    TestCase("FIND AI DISCUSSIONS", "action", "edge_case", "medium", "All caps"),
    TestCase("find ai stuff", "action", "edge_case", "medium", "Casual language"),
    TestCase("yo, search for some tech stuff", "action", "edge_case", "medium", "Very casual language"),
    
    # === CONTEXT-DEPENDENT CASES ===
    TestCase("more on this topic", "action", "context_dependent", "hard", "Needs conversation context"),
    TestCase("what else?", "action", "context_dependent", "hard", "Follow-up question"),
    TestCase("continue", "action", "context_dependent", "hard", "Continuation request"),
    TestCase("next", "action", "context_dependent", "hard", "Next request"),
    TestCase("similar threads", "action", "context_dependent", "hard", "Related content request"),
    
    # === CONVERSATIONAL VARIATIONS ===
    TestCase("I'm curious about your thoughts on AI", "chat", "opinion_request", "medium", "Polite opinion request"),
    TestCase("could you walk me through how React works?", "chat", "explanation_request", "medium", "Tutorial request"),
    TestCase("I'd love to learn more about startups", "chat", "learning_intent", "medium", "Learning expression"),
    TestCase("what would you say about machine learning?", "chat", "opinion_request", "medium", "Opinion question variant"),
    TestCase("do you have any insights on blockchain?", "chat", "opinion_request", "medium", "Insights request"),
    
    # === SEARCH VARIATIONS ===
    TestCase("any recent threads on AI?", "action", "temporal_search", "medium", "Time-based search"),
    TestCase("popular posts about React?", "action", "quality_search", "medium", "Quality-filtered search"),
    TestCase("controversial discussions on crypto?", "action", "quality_search", "medium", "Controversy search"),
    TestCase("beginner-friendly posts on JavaScript?", "action", "quality_search", "medium", "Level-specific search"),
    TestCase("deep technical discussions on ML?", "action", "quality_search", "medium", "Depth-specific search"),
    
    # === FALSE POSITIVES (should be CHAT but might be classified as ACTION) ===
    TestCase("what do you think I should search for?", "chat", "meta_search", "hard", "Meta-question about searching"),
    TestCase("how do I search effectively?", "chat", "meta_search", "medium", "Question about search process"),
    TestCase("should I look for React tutorials?", "chat", "advice_request", "medium", "Seeking advice about searching"),
    TestCase("is it worth searching for AI discussions?", "chat", "advice_request", "medium", "Value judgment question"),
    
    # === COMMAND VARIATIONS ===
    TestCase("retrieve posts about startups", "action", "formal_search", "medium", "Formal search command"),
    TestCase("fetch discussions on blockchain", "action", "formal_search", "medium", "Technical search verb"),
    TestCase("pull up threads about Python", "action", "casual_search", "medium", "Casual search phrasing"),
    TestCase("bring up posts on remote work", "action", "casual_search", "medium", "Bring up variant"),
    TestCase("surface discussions about web3", "action", "formal_search", "medium", "Surface as search verb")
]

DEFAULT_MODELS = [
    "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    "DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC"
]

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
        # Fallback prompt
        return """You are an intelligent assistant that determines if a user message is asking to search for something.

Analyze this message and determine:
1. Is the user asking to search for, find, or look up information?
2. If yes, what specific search query should be extracted?
3. How confident are you in this assessment?

User message: "{message}"

Respond with ONLY valid JSON in this exact format:
{{
  "isSearch": boolean,
  "searchQuery": "extracted search terms or null",
  "confidence": number between 0 and 1,
  "reasoning": "brief explanation"
}}

Examples:
- "find discussions about AI" → {{"isSearch": true, "searchQuery": "AI", "confidence": 0.9, "reasoning": "Clear search intent with specific topic"}}
- "what about startups?" → {{"isSearch": true, "searchQuery": "startups", "confidence": 0.8, "reasoning": "Question about a topic implies search intent"}}
- "hello how are you" → {{"isSearch": false, "searchQuery": null, "confidence": 0.9, "reasoning": "Greeting with no search intent"}}
- "can you help me understand React" → {{"isSearch": false, "searchQuery": null, "confidence": 0.7, "reasoning": "Asking for explanation, not search"}}

JSON Response:"""

class IntentEvaluator:
    def __init__(self, model_name: str = None):
        self.engine = None
        self.model_path = None
        self.model_name = model_name or DEFAULT_MODELS[0]
        self.prompt_template = load_prompt_from_typescript()
        self.results: List[EvalResult] = []
        self.find_model_path()
    
    def find_model_path(self):
        """Find the local model path"""
        current_dir = Path.cwd()
        model_dir = current_dir / "models" / self.model_name
        
        if model_dir.exists():
            self.model_path = str(model_dir)
            return
            
        script_dir = Path(__file__).parent
        if script_dir.name == "mlc_llm":
            repo_root = script_dir.parent
            model_dir = repo_root / "models" / self.model_name
            
            if model_dir.exists():
                self.model_path = str(model_dir)
                return
        
        print(f"❌ Model not found. Make sure models/{self.model_name} exists.")
        sys.exit(1)
    
    def init_engine(self):
        """Initialize the MLC engine"""
        if not MLC_AVAILABLE:
            print("❌ MLC LLM not available")
            return False
            
        if self.engine is None:
            print("🚀 Initializing MLC Engine...")
            try:
                self.engine = MLCEngine(self.model_path)
                print("✅ Engine initialized successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize engine: {e}")
                return False
        return True
    
    def call_model(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        """Call the model with the given prompt"""
        if not self.init_engine():
            return None
            
        try:
            response = self.engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,  # Increased for reasoning models
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Model call failed: {e}")
            return None
    
    def parse_response(self, response: str) -> Dict:
        """Parse JSON response from model"""
        if not response:
            return {"error": "No response from model"}
            
        try:
            # Handle reasoning models with <think> tags
            cleaned_response = response
            if '<think>' in response:
                # Remove thinking blocks
                cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            
            # Extract JSON from response - improved regex for nested structures
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response)
            if not json_match:
                # Fallback: try to find any JSON-like structure
                json_match = re.search(r'\{.*?\}', cleaned_response, flags=re.DOTALL)
                if not json_match:
                    return {"error": "No JSON found in response", "raw": response}
            
            json_text = json_match.group().strip()
            parsed = json.loads(json_text)
            
            if "isSearch" not in parsed or not isinstance(parsed["isSearch"], bool):
                return {"error": "Invalid isSearch field", "raw": response}
            
            if "confidence" not in parsed or not (0 <= parsed["confidence"] <= 1):
                return {"error": "Invalid confidence field", "raw": response}
            
            # Convert to standard format
            parsed["intentCategory"] = "action" if parsed["isSearch"] else "chat"
            return parsed
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": response}
        except Exception as e:
            return {"error": f"Parse error: {e}", "raw": response}
    
    def evaluate_test_case(self, test_case: TestCase, temperature: float = 0.1, verbose: bool = True) -> Optional[EvalResult]:
        """Evaluate a single test case"""
        if verbose:
            print(f"\n🧪 Testing: \"{test_case.query}\" ({test_case.category}, {test_case.difficulty})")
        
        # Build prompt
        prompt = self.prompt_template.replace('{message}', test_case.query)
        
        # Call model
        response = self.call_model(prompt, temperature)
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
        correct = predicted == test_case.expected
        
        result = EvalResult(
            query=test_case.query,
            expected=test_case.expected,
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
                print(f"   Expected: {test_case.expected}")
                print(f"   Reasoning: {result.reasoning}")
        
        return result
    
    def run_full_evaluation(self, temperature: float = 0.1, dataset_filter: str = None, verbose: bool = True) -> List[EvalResult]:
        """Run evaluation on all or filtered test cases"""
        test_cases = COMPREHENSIVE_TEST_CASES
        
        if dataset_filter:
            test_cases = [tc for tc in test_cases if dataset_filter in tc.category]
            print(f"📊 Running evaluation on {len(test_cases)} test cases (filter: {dataset_filter})")
        else:
            print(f"📊 Running full evaluation on {len(test_cases)} test cases")
        
        results = []
        
        # Use tqdm if available, otherwise fallback to manual progress
        if TQDM_AVAILABLE and not verbose:
            iterator = tqdm(test_cases, desc=f"Evaluating {self.model_name}", unit="test")
        else:
            iterator = test_cases
        
        for i, test_case in enumerate(iterator, 1):
            if verbose:
                print(f"\nProgress: {i}/{len(test_cases)}")
            elif not TQDM_AVAILABLE:
                # Show progress even in quiet mode for comparisons when no tqdm
                if i % 10 == 0 or i == len(test_cases):
                    print(f"📊 Progress: {i}/{len(test_cases)} ({i/len(test_cases)*100:.0f}%)")
            
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
        true_positive = sum(1 for r in results if r.expected == "action" and r.predicted == "action")
        false_positive = sum(1 for r in results if r.expected == "chat" and r.predicted == "action")
        true_negative = sum(1 for r in results if r.expected == "chat" and r.predicted == "chat")
        false_negative = sum(1 for r in results if r.expected == "action" and r.predicted == "chat")
        
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
        false_positives = [r for r in failures if r.expected == "chat" and r.predicted == "action"]
        false_negatives = [r for r in failures if r.expected == "action" and r.predicted == "chat"]
        
        print(f"\n❌ FALSE POSITIVES (classified as action, should be chat): {len(false_positives)}")
        for fp in false_positives[:10]:  # Show top 10
            print(f"   \"{fp.query}\" → {fp.predicted} (conf: {fp.confidence:.2f})")
            print(f"      Reasoning: {fp.reasoning}")
            print(f"      Category: {fp.category}, Notes: {fp.notes}")
            print()
        
        print(f"\n❌ FALSE NEGATIVES (classified as chat, should be action): {len(false_negatives)}")
        for fn in false_negatives[:10]:  # Show top 10
            print(f"   \"{fn.query}\" → {fn.predicted} (conf: {fn.confidence:.2f})")
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
    
    def export_results(self, filename: str, results: List[EvalResult] = None) -> None:
        """Export results to JSON or CSV file"""
        if results is None:
            results = self.results
        
        filepath = Path(filename)
        
        if filepath.suffix.lower() == '.json':
            # Export to JSON
            export_data = {
                "metadata": {
                    "total_cases": len(results),
                    "model": self.model_name,
                    "timestamp": str(Path(__file__).stat().st_mtime)
                },
                "metrics": self.calculate_metrics(results),
                "results": [asdict(r) for r in results]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📄 Results exported to {filepath}")
            
        elif filepath.suffix.lower() == '.csv':
            # Export to CSV
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['query', 'expected', 'predicted', 'confidence', 'correct', 'category', 'difficulty', 'reasoning', 'notes', 'model'])
                for r in results:
                    writer.writerow([r.query, r.expected, r.predicted, r.confidence, r.correct, r.category, r.difficulty, r.reasoning, r.notes, r.model_name])
            print(f"📄 Results exported to {filepath}")
        else:
            print(f"❌ Unsupported file format: {filepath.suffix}")

def compare_models(model_a: str, model_b: str, temperature: float = 0.1, dataset_filter: str = None, verbose: bool = False) -> ModelComparison:
    """Compare two models sequentially to avoid memory issues"""
    print(f"🔬 COMPARING MODELS: {model_a} vs {model_b}")
    print("=" * 60)
    
    # Evaluate model A
    print(f"\n📊 Evaluating {model_a}...")
    print(f"🚀 Loading {model_a}...")
    evaluator_a = IntentEvaluator(model_a)
    results_a = evaluator_a.run_full_evaluation(temperature, dataset_filter, verbose=False)
    metrics_a = evaluator_a.calculate_metrics(results_a)
    
    # 🧹 Clean up model A from memory
    del evaluator_a
    import gc
    gc.collect()
    print(f"✅ {model_a} evaluation complete, memory freed")
    
    # Evaluate model B  
    print(f"\n📊 Evaluating {model_b}...")
    print(f"🚀 Loading {model_b}...")
    evaluator_b = IntentEvaluator(model_b)
    results_b = evaluator_b.run_full_evaluation(temperature, dataset_filter, verbose=False)
    metrics_b = evaluator_b.calculate_metrics(results_b)
    
    # 🧹 Clean up model B from memory
    del evaluator_b
    gc.collect()
    print(f"✅ Model B evaluation complete, memory freed")
    
    # Find disagreements
    disagreements = []
    for ra, rb in zip(results_a, results_b):
        if ra.query == rb.query and ra.predicted != rb.predicted:
            disagreements.append((ra.query, ra, rb))
    
    comparison = ModelComparison(
        model_a=model_a,
        model_b=model_b,
        results_a=results_a,
        results_b=results_b,
        accuracy_a=metrics_a.get('accuracy', 0),
        accuracy_b=metrics_b.get('accuracy', 0),
        disagreements=disagreements
    )
    
    return comparison

def print_comparison_report(comparison: ModelComparison) -> None:
    """Print comprehensive model comparison report"""
    print(f"\n🏆 MODEL COMPARISON REPORT")
    print("=" * 50)
    
    # Overall comparison
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   {comparison.model_a:30s}: {comparison.accuracy_a:5.1%}")
    print(f"   {comparison.model_b:30s}: {comparison.accuracy_b:5.1%}")
    
    winner = comparison.model_a if comparison.accuracy_a > comparison.accuracy_b else comparison.model_b
    diff = abs(comparison.accuracy_a - comparison.accuracy_b)
    print(f"   🏅 Winner: {winner} (+{diff:.1%})")
    
    # Disagreement analysis
    print(f"\n🤔 DISAGREEMENTS: {len(comparison.disagreements)} cases")
    if comparison.disagreements:
        print("   Top disagreements:")
        for i, (query, result_a, result_b) in enumerate(comparison.disagreements[:10]):
            print(f"\n   {i+1}. \"{query}\"")
            print(f"      Expected: {result_a.expected}")
            print(f"      {comparison.model_a}: {result_a.predicted} (conf: {result_a.confidence:.2f})")
            print(f"      {comparison.model_b}: {result_b.predicted} (conf: {result_b.confidence:.2f})")
            
            # Show which model was correct
            a_correct = result_a.correct
            b_correct = result_b.correct
            if a_correct and not b_correct:
                print(f"      ✅ {comparison.model_a} correct, {comparison.model_b} wrong")
            elif b_correct and not a_correct:
                print(f"      ✅ {comparison.model_b} correct, {comparison.model_a} wrong")
            elif not a_correct and not b_correct:
                print(f"      ❌ Both models wrong")
    
    # Category analysis
    print(f"\n📂 PERFORMANCE BY CATEGORY:")
    category_stats_a = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stats_b = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result in comparison.results_a:
        category_stats_a[result.category]["total"] += 1
        if result.correct:
            category_stats_a[result.category]["correct"] += 1
    
    for result in comparison.results_b:
        category_stats_b[result.category]["total"] += 1
        if result.correct:
            category_stats_b[result.category]["correct"] += 1
    
    all_categories = set(category_stats_a.keys()) | set(category_stats_b.keys())
    for category in sorted(all_categories):
        acc_a = category_stats_a[category]["correct"] / category_stats_a[category]["total"] if category_stats_a[category]["total"] > 0 else 0
        acc_b = category_stats_b[category]["correct"] / category_stats_b[category]["total"] if category_stats_b[category]["total"] > 0 else 0
        
        better = "A" if acc_a > acc_b else "B" if acc_b > acc_a else "="
        print(f"   {category:20s}: {acc_a:5.1%} vs {acc_b:5.1%} ({better})")

def export_comparison(comparison: ModelComparison, filename: str) -> None:
    """Export model comparison to file"""
    filepath = Path(filename)
    
    if filepath.suffix.lower() == '.json':
        export_data = {
            "metadata": {
                "model_a": comparison.model_a,
                "model_b": comparison.model_b,
                "accuracy_a": comparison.accuracy_a,
                "accuracy_b": comparison.accuracy_b,
                "disagreements": len(comparison.disagreements)
            },
            "disagreements": [
                {
                    "query": query,
                    "expected": ra.expected,
                    "model_a_prediction": ra.predicted,
                    "model_a_confidence": ra.confidence,
                    "model_b_prediction": rb.predicted, 
                    "model_b_confidence": rb.confidence,
                    "model_a_correct": ra.correct,
                    "model_b_correct": rb.correct
                }
                for query, ra, rb in comparison.disagreements
            ],
            "results_a": [asdict(r) for r in comparison.results_a],
            "results_b": [asdict(r) for r in comparison.results_b]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"📄 Comparison exported to {filepath}")
    else:
        print(f"❌ Unsupported format for comparison export: {filepath.suffix}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive intent detection evaluation')
    parser.add_argument('--full-eval', action='store_true', help='Run full evaluation on all test cases')
    parser.add_argument('--dataset', help='Filter test cases by category (e.g., "ambiguous", "edge_case")')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--analyze-failures', action='store_true', help='Show detailed failure analysis')
    parser.add_argument('--export-results', help='Export results to file (.json or .csv)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--model', help=f'Model to use (default: {DEFAULT_MODELS[0]})')
    parser.add_argument('--compare', nargs=2, metavar=('MODEL_A', 'MODEL_B'), help='Compare two models side-by-side')
    parser.add_argument('--export-comparison', help='Export comparison results to file (.json)')
    
    args = parser.parse_args()
    
    # Handle model comparison
    if args.compare:
        try:
            comparison = compare_models(
                args.compare[0], 
                args.compare[1], 
                temperature=args.temp,
                dataset_filter=args.dataset,
                verbose=not args.quiet
            )
            
            print_comparison_report(comparison)
            
            if args.export_comparison:
                export_comparison(comparison, args.export_comparison)
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    
    # Single model evaluation
    if not any([args.full_eval, args.dataset, args.analyze_failures]):
        parser.print_help()
        print(f"\nExamples:")
        print(f"  python mlc_llm/eval_intent_detection.py --full-eval")
        print(f"  python mlc_llm/eval_intent_detection.py --dataset ambiguous --temp 0.2")
        print(f"  python mlc_llm/eval_intent_detection.py --full-eval --export-results results.json")
        print(f"  python mlc_llm/eval_intent_detection.py --compare {DEFAULT_MODELS[0]} {DEFAULT_MODELS[1]}")
        return
    
    model_name = args.model or DEFAULT_MODELS[0]
    evaluator = IntentEvaluator(model_name)
    
    try:
        if args.full_eval or args.dataset:
            # Run evaluation
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