#!/usr/bin/env python3
"""
MLC Model Wrapper

Core model inference functionality that can be reused across different evaluation scripts.
Handles model loading, inference, and response parsing.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import re
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
            response = self.engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Model call failed: {e}")
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