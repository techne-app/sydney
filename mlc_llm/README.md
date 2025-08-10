# Intent Detection Model Evaluation

This directory contains tools for evaluating and testing intent detection models used in the Techne browser extension.

## Overview

The intent detection system uses a **two-phase classification approach** to route user requests in the chat interface:

### Phase 1: Chat vs Action Classification
Determines whether user messages are asking for **CONVERSATIONAL** responses or want the system to **DO SOMETHING** (take action).

- **chat**: User wants conversation, explanations, opinions, advice, or general discussion
- **action**: User wants the system to perform an action - search, find, retrieve, create, analyze, etc.

### Phase 2: Function Calling (Action Cases Only)
For messages classified as "action", determines which specific function to call and extracts parameters:

- **get_thread_cards**: Find, show, retrieve, discover HN content/discussions using real backend API
- **create_summary**: Generate summaries of discussion threads (future) 
- **analyze_trends**: Analyze trends, compare, synthesize information (future)

This two-phase approach is **extensible and MCP-ready** - new functions can be added without retraining Phase 1, and the architecture naturally maps to tool calling patterns for MCP server integration.

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

### `eval_intent_detection.py`
Comprehensive evaluation framework using BFCL (Berkeley Function Calling Leaderboard) format natively.

```bash
# Full evaluation with default model
python mlc_llm/eval_intent_detection.py --full-eval

# Compare two models
python mlc_llm/eval_intent_detection.py --compare "Phi-3.5-mini-instruct-q4f16_1-MLC" "Llama-3.2-3B-Instruct-q4f16_1-MLC"

# Test specific dataset categories
python mlc_llm/eval_intent_detection.py --dataset explicit_search
python mlc_llm/eval_intent_detection.py --dataset ambiguous

# Failure analysis
python mlc_llm/eval_intent_detection.py --analyze-failures

# Export results
python mlc_llm/eval_intent_detection.py --export-results results.json
```

## Model Evaluation Results

### Phase 1: Chat vs Action Classification (138 Test Cases)

| Model | Size (MB) | Accuracy | Precision | Recall | F1 Score | Production Ready | 
|-------|-----------|----------|-----------|--------|----------|------------------|
| 🥇 **Phi-3.5-mini-instruct** | 2,052 | **87.0%** | **93.2%** | **84.1%** | **0.885** | ✅ **EXCELLENT** |
| 🥈 **Llama-3.2-3B-Instruct** | 1,733 | **85.8%** | **86.9%** | **90.1%** | **0.885** | ✅ **EXCELLENT** |
| 🥉 **Gemma-2-2B-it** | 1,420 | **76.8%** | **74.0%** | **93.9%** | **0.828** | ⚠️ **ACCEPTABLE** |

### Multi-Run Accuracy Analysis 📊
**Accuracy across five evaluation runs (138 test cases each):**

| Model | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Std Dev |
|-------|-------|-------|-------|-------|-------|------|---------|
| **Phi-3.5-mini-instruct** | 85.9% | 87.0% | 87.0% | 87.7% | **87.0%** | **86.9%** | **±0.7%** |
| **Llama-3.2-3B-Instruct** | 85.9% | 87.4% | 88.9% | 85.9% | **85.8%** | **86.8%** | **±1.2%** |
| **Gemma-2-2B-it** | 76.1% | 76.1% | 76.1% | 76.1% | **76.8%** | **76.2%** | **±0.3%** |

### Key Findings

#### 🏆 Phi-3.5-mini-instruct (RECOMMENDED)
- **Consistent high accuracy** averaging 86.9% across 5 runs with latest at 87.0%
- **Exceptional precision** at 93.2% - makes very few false positive errors
- **Most consistent** performance across runs (±0.7% variance) - improved stability
- **Best for applications** where false positives (classifying chat as search) are costly

#### ✅ Llama-3.2-3B-Instruct (ALTERNATIVE)
- **Stable performance** averaging 86.8% across 5 runs with latest at 85.8%
- **Excellent recall** at 90.1% - catches more search intents 
- **Smaller model** at 1.7GB - good balance of performance and efficiency
- **Moderate variance** (±1.2%) - reasonably predictable performance

## Test Dataset Structure (BFCL Format)

The evaluation system now uses **BFCL (Berkeley Function Calling Leaderboard) compatible format** for comprehensive multi-phase evaluation:

### Current Dataset: `bfcl_testcases.json`

```json
{
  "metadata": {
    "description": "Intent detection test cases using real get_thread_cards API for MCP evaluation",
    "version": "1.0.0",
    "format": "Berkeley Function Calling Leaderboard (BFCL) compatible",
    "total_cases": 138,
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
3. Local LLM processes message with `SEARCH_INTENT_PROMPT`
4. JSON response parsed for `isSearch`, `searchQuery`, `confidence`
5. Router directs to search service or conversational AI

### Prompt Engineering
The shared prompt (`src/prompts/searchIntent.ts`) includes:
- **Clear intent definitions** with examples
- **Social interaction patterns** for chat classification  
- **Ambiguous case handling** with confidence scores
- **Structured JSON output** specification

## Prerequisites

### Required Dependencies
The evaluation scripts now require explicit dependencies (no fallbacks):

- **tqdm**: Progress bar library
- **MLC LLM**: Model inference engine
- **TestCaseLoader**: BFCL format parser

```bash
# Install required dependencies
pip install tqdm
uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly
uv pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly
```

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
The evaluation script requires the actual prompt from the TypeScript source:
- Must have `src/prompts/searchIntent.ts` with `SEARCH_INTENT_PROMPT`
- No fallback prompts - script exits if prompt cannot be loaded
- Ensures evaluation uses identical prompt as production system

## Production Configuration

### Current Extension Settings
```typescript
// src/config.ts
DEFAULT_MODEL: "Llama-3.2-3B-Instruct-q4f16_1-MLC"  // UPDATE RECOMMENDED

// Recommended update:
DEFAULT_MODEL: "Phi-3.5-mini-instruct-q4f16_1-MLC"
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
2. Run smoke test with `python mlc_llm/model_wrapper.py --model <model-name>`
3. Run full evaluation with `eval_intent_detection.py`
4. Update configuration if performance improves

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