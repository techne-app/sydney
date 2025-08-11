#!/usr/bin/env python3
"""
MLC Model Wrapper

Core model inference functionality that can be reused across different evaluation scripts.
Handles model loading, inference, and response parsing with smoke testing capabilities.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
import time
import json

try:
    from mlc_llm import MLCEngine
    MLC_AVAILABLE = True
except ImportError:
    print("⚠️  MLC LLM not installed. Install with:")
    print("   uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly")
    print("   uv pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly")
    MLC_AVAILABLE = False

class MLCModelWrapper:
    """Wrapper for MLC LLM models with common inference functionality"""
    
    def __init__(self, model_name: str):
        if not MLC_AVAILABLE:
            raise ImportError("MLC LLM not installed")
            
        self.model_name = model_name
        self.engine = None
        self.model_path = None
        self.model_size_mb = 0.0
        self.last_inference_time = 0.0  # Track timing of last inference
        self._find_model_path()
        self._calculate_model_size()
    
    def _find_model_path(self):
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
    
    def _calculate_model_size(self):
        """Calculate total model size in MB"""
        if not self.model_path:
            return
            
        try:
            total_size = 0
            model_dir = Path(self.model_path)
            
            # Sum up all .bin files (model weights)
            for bin_file in model_dir.glob('*.bin'):
                total_size += bin_file.stat().st_size
            
            # Also include tokenizer and config files
            for other_file in model_dir.glob('*.json'):
                total_size += other_file.stat().st_size
            
            # Convert to MB
            self.model_size_mb = total_size / (1024 * 1024)
            
        except Exception as e:
            print(f"⚠️  Could not calculate model size: {e}")
            self.model_size_mb = 0.0
    
    def init_engine(self):
        """Initialize the MLC engine"""
        if self.engine is None:
            print(f"🚀 Loading model: {self.model_name}")
            try:
                self.engine = MLCEngine(self.model_path)
                print(f"✅ Model loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to load model {self.model_name}: {e}")
                return False
        return True
    
    def call_model(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> Optional[str]:
        """Call the model with the given prompt"""
        if not self.init_engine():
            return None
            
        try:
            # Measure only the actual inference time (exclude initialization)
            
            start_time = time.time()
            response = self.engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            self.last_inference_time = time.time() - start_time
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Model call failed: {e}")
            self.last_inference_time = 0.0
            return None
    
    def call_model_with_system(self, system_prompt: str, user_prompt: str, 
                              temperature: float = 0.1, max_tokens: int = 500) -> Optional[str]:
        """Call the model with separate system and user prompts"""
        if not self.init_engine():
            return None
            
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.engine.chat.completions.create(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Model call failed: {e}")
            return None
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from model with error handling"""
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
            return parsed
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": response}
        except Exception as e:
            return {"error": f"Parse error: {e}", "raw": response}
    
    def cleanup(self):
        """Clean up model resources"""
        if self.engine:
            del self.engine
            self.engine = None
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "path": self.model_path,
            "size_mb": self.model_size_mb
        }
    
    def load_prompt_from_typescript(self) -> str:
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
    
    def get_expected_result(self, query: str) -> str:
        """Simple heuristic to determine expected result for smoke testing"""
        action_keywords = ['find', 'search', 'show', 'get', 'look up', 'retrieve']
        query_lower = query.lower()
        
        for keyword in action_keywords:
            if keyword in query_lower:
                return 'action'
        return 'chat'
    
    def smoke_test(self, test_queries: Optional[List[str]] = None, temperature: float = 0.1, verbose: bool = True) -> bool:
        """
        Run a simple smoke test to verify model functionality before full evaluation.
        
        Args:
            test_queries: Optional list of test queries. Uses default set if None.
            temperature: Model temperature for testing
            verbose: Whether to print detailed output
            
        Returns:
            True if smoke test passes, False otherwise
        """
        if test_queries is None:
            test_queries = [
                "find discussions about AI",  # Should be action
                "hello how are you",          # Should be chat
                "search for startups",        # Should be action 
                "what do you think about React?"  # Should be chat
            ]
        
        if verbose:
            print(f"\n🧪 SMOKE TEST: {self.model_name}")
            print("=" * 50)
        
        try:
            # Load prompt template
            prompt_template = self.load_prompt_from_typescript()
            
            passed = 0
            total = len(test_queries)
            
            for i, query in enumerate(test_queries, 1):
                if verbose:
                    print(f"\n[{i}/{total}] Testing: \"{query}\"")
                
                # Build prompt
                prompt = prompt_template.replace('{message}', query)
                
                # Call model
                response = self.call_model(prompt, temperature, max_tokens=500)
                if not response:
                    if verbose:
                        print("   ❌ No response from model")
                    continue
                
                # Parse response
                parsed = self.parse_json_response(response)
                
                if "error" in parsed:
                    if verbose:
                        print(f"   ❌ Parse Error: {parsed['error']}")
                    continue
                
                # Validate response structure for intent detection
                if "isSearch" not in parsed or not isinstance(parsed["isSearch"], bool):
                    if verbose:
                        print("   ❌ Missing or invalid isSearch field")
                    continue
                
                if "confidence" not in parsed or not (0 <= parsed["confidence"] <= 1):
                    if verbose:
                        print("   ❌ Missing or invalid confidence field")
                    continue
                
                # Check correctness
                predicted = "action" if parsed["isSearch"] else "chat"
                expected = self.get_expected_result(query)
                correct = predicted == expected
                
                if correct:
                    passed += 1
                
                if verbose:
                    status = "✅ PASS" if correct else "❌ FAIL"
                    print(f"   Predicted: {predicted} (confidence: {parsed.get('confidence', 0):.2f}) → {status}")
                    if not correct:
                        print(f"   Expected: {expected}")
            
            success = passed == total
            
            if verbose:
                print(f"\n🎯 SMOKE TEST RESULT: {passed}/{total} passed")
                if success:
                    print("✅ Model is ready for evaluation")
                else:
                    print("❌ Model failed smoke test - check configuration")
            
            return success
            
        except Exception as e:
            if verbose:
                print(f"❌ Smoke test failed with error: {e}")
            return False


def main():
    """CLI interface for smoke testing models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smoke test MLC models for intent detection')
    parser.add_argument('--model', default='Phi-3.5-mini-instruct-q4f16_1-MLC', help='Model to test')
    parser.add_argument('--query', help='Test a single query')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature (0.0-1.0)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        wrapper = MLCModelWrapper(args.model)
        
        if args.query:
            # Test a single query
            test_queries = [args.query]
        else:
            # Use default smoke test queries
            test_queries = None
        
        success = wrapper.smoke_test(test_queries, args.temp, verbose=not args.quiet)
        wrapper.cleanup()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()