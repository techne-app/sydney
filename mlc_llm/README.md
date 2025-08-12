# Intent Detection Model Evaluation

This directory contains tools for evaluating and testing intent detection models used in the Techne browser extension.

## Overview

The intent detection system uses a **two-step classification approach** to route user requests in the chat interface:

### Step 1: Intent Classification
Determines whether user messages are asking for **CONVERSATIONAL** responses or want the system to **TAKE ACTION**.

- **chat**: User wants conversation, explanations, opinions, advice, or general discussion
- **action**: User wants the system to perform an action - search, find, retrieve, create, analyze, etc.

### Step 2: Function Calling (Action Cases Only)
For messages classified as "action", determines which specific function to call and extracts parameters:

- **get_thread_cards**: Find, show, retrieve, discover HN content/discussions using real backend API
- **summarize_pinned_thread**: Generate summaries of pinned discussion threads
- **no_action**: Fallback for conversational responses

### Two-Step Evaluation Methodology
The evaluation system separately measures performance at each step:

- **Step 1 Accuracy**: Percentage of test cases where intent classification (chat vs action) is correct
- **Step 2 Accuracy**: Percentage of action cases where function calling and parameter extraction is correct  
- **Overall Accuracy**: Combined accuracy across both steps - a test case must pass both steps to be considered correct

This approach provides granular insights into model performance and identifies whether failures occur during intent detection or function calling. It's **extensible and MCP-ready** - new functions can be added without retraining Step 1, and the architecture naturally maps to tool calling patterns for MCP server integration.

## Scripts

### `model_wrapper.py`
Core model inference wrapper with built-in smoke testing functionality for quick validation before full evaluation.

```bash
# Run default smoke test (4 basic queries)
python mlc_llm/model_wrapper.py

# Test specific model
python mlc_llm/model_wrapper.py --model "Phi-3.5-mini-instruct-q4f16_1-MLC"

# Test a single query
python mlc_llm/model_wrapper.py --query "find AI discussions"

# Adjust temperature
python mlc_llm/model_wrapper.py --temp 0.3 --query "search for startups"

# Quiet mode (less verbose output)
python mlc_llm/model_wrapper.py --quiet
```

The smoke test validates basic model functionality by testing 4 representative queries (2 action, 2 chat) and verifies that the model loads correctly, produces valid JSON responses, and achieves reasonable accuracy on simple cases. It's designed as a quick sanity check before running full evaluations.

### `eval_two_step.py`
Comprehensive two-step evaluation framework that separately evaluates intent classification (Step 1) and function calling (Step 2).

```bash
# Full evaluation with default settings (5 iterations)
uv run python mlc_llm/eval_two_step.py

# Extended evaluation (10 iterations for more stable results)
uv run python mlc_llm/eval_two_step.py --iterations 10

# Quiet mode (less verbose output)
uv run python mlc_llm/eval_two_step.py --iterations 10 --quiet

# Test specific models
uv run python mlc_llm/eval_two_step.py --model "Phi-3.5-mini-instruct-q4f16_1-MLC"

# Compare multiple models (automatically detected from models/ directory)
uv run python mlc_llm/eval_two_step.py --iterations 10 --quiet
```

## Model Evaluation Results

### Latest Two-Step Evaluation (158 Test Cases, 10 Iterations)

🏆 **TWO-STEP MODEL COMPARISON TABLE**
========================================================================================================================
| Model                     | Size (MB) | Step1 Acc | Step2 Acc | Overall | Step1 Time | Step2 Time | Status         |
|---------------------------|-----------|-----------|-----------|---------|------------|------------|----------------|
| **Phi-3.5-mini-instruct** | 2,052    | **81.7%** | **87.2%** | **77.3%** | 1.675s     | 3.215s     | ⚠️ **ACCEPTABLE** |
| **gemma-2-2b-it**         | 1,420    | 79.7%     | 69.8%     | 73.4%   | 1.073s     | 1.768s     | ❌ **POOR**       |
| **Llama-3.2-3B-Instruct** | 1,733    | 71.1%     | 75.2%     | 70.8%   | 0.772s     | 1.695s     | ❌ **POOR**       |

### 📈 Two-Step Performance Summary

- **Best Step 1 Accuracy**: 81.7% (Phi-3.5-mini-instruct)
- **Best Step 2 Accuracy**: 87.2% (Phi-3.5-mini-instruct) 
- **Best Overall Accuracy**: 77.3% (Phi-3.5-mini-instruct)
- **Average Step 1**: 77.5%
- **Average Step 2**: 77.4%
- **Average Overall**: 73.8%

### 💡 Key Findings & Recommendations

#### 🏆 Phi-3.5-mini-instruct (BEST OVERALL)
- **Highest overall accuracy** at 77.3% - best combined two-step performance
- **Excellent Step 1 (Intent)** at 81.7% - best at distinguishing chat vs action
- **Outstanding Step 2 (Function)** at 87.2% - superior function calling accuracy
- **Larger model** at 2,052 MB but justified by superior accuracy
- **Slower inference** but provides the most reliable intent detection
- **Production recommended** - clear leader for two-step evaluation

#### 🔬 gemma-2-2b-it (COMPACT BUT INCONSISTENT)
- **Smallest model** at 1,420 MB - best resource efficiency
- **Good Step 1 performance** at 79.7% - decent intent classification
- **Poor Step 2 performance** at 69.8% - struggles with function calling
- **Fastest total time** but inconsistent accuracy across steps
- **Not recommended** - unreliable for production use

#### 📱 Llama-3.2-3B-Instruct (UNDERPERFORMING)
- **Previously strong single-step** but struggles with two-step approach
- **Weak Step 1** at 71.1% - poor intent classification
- **Mediocre Step 2** at 75.2% - average function calling
- **Fast inference** but accuracy too low for reliable production use
- **Not recommended** - significant accuracy regression in two-step evaluation

## Test Dataset Structure (BFCL Format)

The evaluation system now uses **BFCL (Berkeley Function Calling Leaderboard) compatible format** for comprehensive multi-phase evaluation:

### Current Dataset: `bfcl_testcases.json`

```json
{
  "metadata": {
    "description": "Intent detection test cases using real get_thread_cards API for MCP evaluation",
    "version": "1.0.0",
    "format": "Berkeley Function Calling Leaderboard (BFCL) compatible",
    "total_cases": 158,
    "evaluation_phases": {
      "intent_classification": "Phase 1: chat vs action (binary classification)",
      "function_calling": "Phase 2: function selection and parameter extraction (action cases only)"
    }
  },
  "functions": {
    "get_thread_cards": {
      "name": "get_thread_cards",
      "description": "Get filtered thread cards from Hacker News discussions",
      "parameters": {
        "type": "object",
        "properties": {
          "keyword_filter": {"type": "string", "description": "Text to filter discussions"},
          "hours_back": {"type": "integer", "description": "Hours to look back", "default": 168},
          "sort_by": {"type": "string", "enum": ["karma_density", "recent", "comment_count"]},
          "num_cards": {"type": "integer", "description": "Number of cards to return"},
          "density_min_comment_constant": {"type": "integer", "description": "Quality threshold"}
        },
        "required": ["keyword_filter"]
      }
    },
    "no_action": {
      "name": "no_action",
      "description": "No action required - conversational response",
      "parameters": {
        "type": "object",
        "properties": {
          "response_type": {"type": "string", "enum": ["greeting", "opinion", "explanation", "social"]}
        },
        "required": ["response_type"]
      }
    }
  },
  "test_cases": [
    {
      "id": "action_001",
      "question": "find discussions about AI",
      "category": "explicit_search",
      "difficulty": "easy",
      "intent_expected": "action",
      "action_type_expected": "get_thread_cards",
      "function": [...],
      "expected_function_call": "get_thread_cards(keyword_filter='AI')",
      "valid_alternatives": ["get_thread_cards(keyword_filter='artificial intelligence')"],
      "evaluation_phases": ["intent_classification", "function_calling"]
    }
  ]
}
```

### Key Improvements in BFCL Format

**1. Real Backend API Integration**
- Uses actual `get_thread_cards` function from Azure Functions backend
- Parameters match real API: `keyword_filter`, `hours_back`, `sort_by`, `num_cards`, `density_min_comment_constant`
- Enables realistic MCP server evaluation

**2. Two-Phase Evaluation Support**
- **Phase 1**: Intent classification (chat vs action) - evaluated on ALL test cases
- **Phase 2**: Function calling with parameter extraction - evaluated on ACTION cases only

**3. Rich Parameter Extraction**
- "recent AI discussions" → `get_thread_cards(keyword_filter="AI", hours_back=168, sort_by="recent")`
- "highly upvoted ML posts" → `get_thread_cards(keyword_filter="ML", density_min_comment_constant=10)`
- "find 5 startup threads" → `get_thread_cards(keyword_filter="startup", num_cards=5)`

## Expanding for Other Evaluation Types

The BFCL format architecture is designed for extensibility. Here's how to add new evaluation types:

### 1. Adding New Function Types

Create new function schemas in the `functions` section:

```json
{
  "functions": {
    "analyze_trends": {
      "name": "analyze_trends",
      "description": "Analyze trends in HN discussions over time",
      "parameters": {
        "type": "object",
        "properties": {
          "topic": {"type": "string", "description": "Topic to analyze"},
          "time_range": {"type": "string", "enum": ["week", "month", "quarter", "year"]},
          "metric": {"type": "string", "enum": ["volume", "sentiment", "engagement"]}
        },
        "required": ["topic"]
      }
    },
    "create_summary": {
      "name": "create_summary",
      "description": "Generate summaries of discussion threads",
      "parameters": {
        "type": "object", 
        "properties": {
          "thread_ids": {"type": "array", "items": {"type": "integer"}},
          "format": {"type": "string", "enum": ["bullet_points", "paragraph", "timeline"]},
          "max_length": {"type": "integer", "default": 500}
        },
        "required": ["thread_ids"]
      }
    }
  }
}
```

### 2. New Evaluation Phases

Add specialized evaluation phases:

```json
{
  "evaluation_phases": {
    "intent_classification": "Phase 1: chat vs action (binary classification)",
    "function_calling": "Phase 2: function selection and parameter extraction", 
    "parameter_validation": "Phase 3: Parameter type and constraint checking (future)",
    "semantic_equivalence": "Phase 4: Alternative parameter matching (future)"
  }
}
```

### 3. Domain-Specific Test Cases

Create specialized test case categories:

```json
{
  "test_cases": [
    {
      "id": "analyze_001",
      "question": "show me how AI discussions evolved this quarter",
      "category": "temporal_analysis",
      "difficulty": "hard",
      "intent_expected": "action",
      "action_type_expected": "analyze_trends",
      "expected_function_call": "analyze_trends(topic='AI', time_range='quarter', metric='volume')",
      "evaluation_phases": ["intent_classification", "function_calling"]
    }
  ]
}
```

### 4. Future Evaluation Framework Extensions

**Multi-Modal Evaluation**
- Image analysis requests
- PDF document processing
- Video content understanding

**Conversational Context**
- Multi-turn conversation evaluation
- Context preservation across turns
- Reference resolution ("find more like that")

**MCP Server Integration**
- Real backend MCP server calls
- Tool orchestration evaluation
- Multi-step reasoning assessment

**Performance Benchmarks**
- Latency under load
- Memory usage profiling
- Concurrent request handling

## Usage in Extension

### Intent Detection Flow
1. User message received in chat interface
2. Message sent to `IntentDetector.detectIntent()`
3. **Step 1**: Local LLM processes message with intent classification prompt
4. **Step 2**: If action detected, second LLM call for function selection and parameter extraction
5. JSON responses parsed for intent, function calls, and confidence scores
6. Router directs to appropriate service (search, chat, or function execution)

### Prompt Engineering
The two-step system uses separate prompts:
- **Step 1 Prompt** (`src/prompts/intentOnly.ts`): Intent classification with clear examples
- **Step 2 Prompt** (`src/prompts/actionOnly.ts`): Function calling with parameter extraction
- **Context awareness** for pinned thread scenarios
- **Structured JSON output** specification for both steps

## Prerequisites

### UV Workflow Setup
This project uses **uv** for Python environment and dependency management. The venv is created in the repo root so VSCode can auto-detect it.

```bash
# Create virtual environment (uses latest available Python)
uv venv --python-preference only-managed

# Or specify a specific Python version (uv will download if needed)
uv venv --python 3.12

# Install required MLC packages (both are required)
# Option 1: Stable versions (CPU-only, recommended for stability)
uv pip install --find-links https://mlc.ai/wheels mlc_llm_cpu==0.19.0 mlc_ai_cpu==0.19.0

# Option 2: Nightly versions (bleeding edge, includes Metal GPU acceleration on Apple Silicon)
uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly
uv pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly

# Install additional dependencies
uv pip install tqdm

# Run scripts using uv (automatically uses the venv)
uv run python mlc_llm/model_wrapper.py
uv run python mlc_llm/eval_intent_detection.py --full-eval

# Or activate venv manually if preferred
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
python mlc_llm/model_wrapper.py
```

### Required Dependencies
The evaluation scripts require these packages:

**Stable versions (CPU-only, recommended for stability):**
- **mlc_llm_cpu==0.19.0**: Core MLC LLM inference engine (stable, CPU-only)
- **mlc_ai_cpu==0.19.0**: Additional AI components (required with mlc-llm)

**Nightly versions (bleeding edge, Metal GPU acceleration):**
- **mlc-llm-nightly**: Core MLC LLM inference engine (latest features, includes Metal GPU support on Apple Silicon)
- **mlc-ai-nightly**: Additional AI components (required with mlc-llm-nightly)

**Additional dependencies:**
- **tqdm**: Progress bar library

### Model Setup
```bash
# Install git-lfs (required for model downloads)
brew install git-lfs
git lfs install

cd models

# Download recommended model
git clone https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC

# Download alternative model
git clone https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC

cd ..
```

### TypeScript Prompt Requirement
The evaluation script requires the actual prompts from the TypeScript source:
- Must have `src/prompts/intentOnly.ts` with intent classification prompt
- Must have `src/prompts/actionOnly.ts` with function calling prompt  
- No fallback prompts - script exits if prompts cannot be loaded
- Ensures evaluation uses identical prompts as production system

## Production Configuration

### Current Extension Settings
```typescript
// src/config.ts
DEFAULT_MODEL: "Phi-3.5-mini-instruct-q4f16_1-MLC"  // ✅ OPTIMAL CHOICE

// Updated based on two-step evaluation results - best overall accuracy (77.3%)
```

### Model Parameters
```typescript
temperature: 0.1,        // Low for consistent JSON output
max_tokens: 500,         // Increased for reasoning models  
stream: false           // Synchronous for intent detection
```

## Development Workflow

### Adding New Test Cases
1. Edit `bfcl_testcases.json` directly
2. Add to appropriate evaluation phases
3. Run evaluation to establish baseline
4. Update production model if needed

### Model Updates
1. Download new model to `models/` directory
2. Run smoke test with `uv run python mlc_llm/model_wrapper.py --model <model-name>`
3. Run full two-step evaluation with `uv run python mlc_llm/eval_two_step.py --iterations 10 --quiet`
4. Update configuration if overall accuracy improves

### Expanding Evaluation Framework
1. Define new functions in BFCL format
2. Create test cases with expected function calls
3. Add evaluation phase support in testcase_loader
4. Update evaluation script for new phases
5. Run comprehensive evaluation

## Future Enhancements

### Planned Improvements
- **Confidence-based routing** with threshold tuning
- **Multi-step intent detection** for complex queries
- **Contextual awareness** using conversation history
- **A/B testing framework** for prompt variations

### MCP Integration Readiness
- **Function calling patterns** prepared for MCP server integration
- **Tool orchestration** architecture for agentic search
- **Multi-step reasoning** capabilities for complex queries
- **Backend connectivity** for historical data access
- **BFCL compatibility** for industry-standard evaluation