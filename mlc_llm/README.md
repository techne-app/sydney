# Intent Detection Model Evaluation

This directory contains tools for evaluating and testing intent detection using models that run on edge devices (e.g. in a browser extension called Techne Navigator).

The core idea here is that we want to use SLMs for determining the user intent and then call the appropriate MCP server. To be sure, the model can do a small task but fundamentally, it needs to be very good at calling for help. 

## Table of Contents

- [Overview](#overview)
  - [Task 1: Intent Classification](#task-1-intent-classification)
  - [Task 2: Function Calling (Action Cases Only)](#task-2-function-calling-action-cases-only)
- [Single-Step Approach](#single-step-approach)
  - [Mechanism](#mechanism)
  - [Evaluation Methodology](#evaluation-methodology)
  - [Advantages](#advantages)
  - [Trade-offs](#trade-offs)
- [Two-Step Approach](#two-step-approach)
  - [Mechanism](#mechanism-1)
  - [Evaluation Methodology](#evaluation-methodology-1)
  - [Advantages](#advantages-1)
  - [Trade-offs](#trade-offs-1)
- [Function Reference](#function-reference)
- [Model Evaluation Results](#model-evaluation-results)
- [Test Dataset Structure](#test-dataset-structure)
- [How to run evaluations](#how-to-run-evaluations)
  - [Setup Environment](#setup-environment)
  - [Download Models](#download-models)
  - [Run Evaluations](#run-evaluations)
  - [Extending Evaluations](#extending-evaluations)
- [Future Enhancements](#future-enhancements)

## Overview

The intent detection system uses Small Language Models (SLMs) to route user requests in the chat interface. Following the principle that models should excel at "calling for help" rather than doing everything themselves, the system focuses on accurately determining user intent and selecting the appropriate MCP server or function to handle the actual task.

The system needs to perform two distinct tasks:

### Task 1: Intent Classification
Determines whether user messages are asking for **CONVERSATIONAL** responses or want the system to **TAKE ACTION**.

- **chat**: User wants conversation, explanations, opinions, advice, or general discussion
- **action**: User wants the system to perform an action - search, find, retrieve, create, analyze, download, upload, etc.

### Task 2: Function Calling (Action Cases Only)
For messages classified as "action", determines which specific function or MCP server to call and extracts the necessary parameters. This embodies the "calling for help" principle - the SLM doesn't perform the actual task but correctly identifies what help is needed and how to request it. See [Function Reference](#function-reference) for detailed specifications.

## Single-Step Approach

The single-step approach combines both intent classification and function calling into a single LLM inference call using a unified prompt.

### Mechanism
- **Unified Processing**: One SLM call processes the user message and returns both intent classification and function selection with parameters
- **Structured Output**: The model returns JSON with intent, function name, parameters, confidence, and reasoning in one response
- **Context Awareness**: Single prompt includes all necessary context for both intent detection and function calling
- **Efficient Routing**: The model focuses on accurately identifying which tool or MCP server can best handle the user's request

### Evaluation Methodology
- **Combined Accuracy**: Test cases are evaluated holistically - both intent and function must be correct to pass
- **Single Inference**: Each test case requires only one model call
- **Prompt**: Uses `src/prompts/singleStep.ts` for unified intent+function detection

### Advantages
- **Faster Inference**: Only one LLM call required instead of two sequential calls
- **Lower Latency**: Reduced total processing time, especially important for edge devices
- **Simpler Architecture**: Single prompt system is easier to maintain and debug
- **Better Context Preservation**: No information loss between separate inference steps

### Trade-offs
- **Complex Prompts**: Single prompt must handle both classification and function calling logic
- **Token Usage**: May require more tokens per inference to handle complex combined logic
- **Less Granular Debugging**: Harder to isolate whether failures occur in intent detection vs function calling

### Prompt
**Single-step prompt**: [src/prompts/singleStep.ts](../src/prompts/singleStep.ts)

This prompt combines intent classification and function calling logic into a unified system that handles both tasks in a single SLM inference call. The prompt is designed to make the model excel at "calling for help" by accurately identifying which tool or MCP server is needed.

## Two-Step Approach

The two-step approach separates intent classification and function calling into sequential LLM inference calls with specialized prompts for each task.

### Mechanism
- **Sequential Processing**: First SLM call determines intent (chat vs action), then second call handles function selection and parameter extraction
- **Specialized Prompts**: Each step uses focused prompts optimized for specific tasks (classification vs function calling)
- **Conditional Flow**: Function calling only occurs if intent is classified as "action"
- **Staged Routing**: Two-stage process for identifying the appropriate tool or MCP server to handle the user's request

### Evaluation Methodology
- **Granular Accuracy**: Test cases are evaluated at each step independently, providing detailed performance insights
- **Step 1 Accuracy**: Percentage of test cases where intent classification (chat vs action) is correct
- **Step 2 Accuracy**: Percentage of action cases where function calling and parameter extraction is correct  
- **Overall Accuracy**: Combined accuracy across both steps - a test case must pass both steps to be considered correct
- **Prompts**: Uses `src/prompts/intentOnly.ts` and `src/prompts/actionOnly.ts`

### Advantages
- **Granular Debugging**: Easy to identify whether failures occur in intent detection vs function calling
- **Specialized Optimization**: Each prompt can be independently optimized for its specific task
- **Modular Architecture**: New functions can be added without modifying intent classification logic
- **Clear Separation**: Distinct evaluation phases provide detailed performance analytics
- **MCP-Ready**: Architecture naturally maps to tool calling patterns for MCP server integration

### Trade-offs
- **Higher Latency**: Two sequential LLM calls increase total processing time
- **Context Loss**: Information may be lost between separate inference steps
- **Complex Orchestration**: Requires coordination between multiple LLM calls
- **Resource Usage**: More compute resources needed for sequential processing

### Prompts
**Step 1 - Intent classification**: [src/prompts/intentOnly.ts](../src/prompts/intentOnly.ts)

**Step 2 - Function calling**: [src/prompts/actionOnly.ts](../src/prompts/actionOnly.ts)

The two-step approach uses specialized prompts: the first determines intent (chat vs action), and if action is detected, the second prompt handles function selection and parameter extraction. Both prompts are optimized to help the SLM excel at "calling for help" by making precise routing decisions.

## Function Reference

The system supports three core functions for handling user requests:

### `search_threads`
**Purpose**: Find, show, retrieve, discover HN content/discussions using real backend API

**Parameters**:
- `keyword_filter` (required): Text to filter discussions  
- `hours_back`: Hours to look back (default: 168)
- `sort_by`: Sort method - "karma_density", "recent", "comment_count"
- `num_cards`: Number of cards to return
- `density_min_comment_constant`: Quality threshold

### `summarize_pinned_thread`  
**Purpose**: Generate summaries of pinned discussion threads

**Parameters**:
- `format`: Summary format - "bullet_points", "paragraph", "timeline"
- `max_length`: Maximum summary length (default: 500)

### `no_action`
**Purpose**: Fallback for conversational responses

**Parameters**:
- `response_type`: Type of response - "greeting", "opinion", "explanation", "social"

## Model Evaluation Results

### Single-Step vs Two-Step Comparison (157 Test Cases, 10 Iterations)

> **Methodology Update**: Results from unified `eval.py` script using 10-iteration statistical sampling for robustness. Function name consistency ensured through single source of truth architecture (dataset-driven evaluation).

#### Unified Evaluation Results

| Model | Single-Step Accuracy | Two-Step Accuracy | Accuracy Difference | Speed Advantage | Recommended |
|-------|---------------------|-------------------|-------------------|-----------------|-------------|
| **Llama-3.2-3B-Instruct** | **80.1%** | 71.6% | **+8.5% single-step** | 2.3x faster | 🟡 Single-Step |
| **gemma-2-2b-it** | 79.2% | 77.7% | +1.5% single-step | 1.6x faster | ⚪ Similar |
| **Phi-3.5-mini-instruct** | 53.5% | **80.6%** | **+27.1% two-step** | 1.2x slower | 🔵 Two-Step |

#### Detailed Breakdown

**Single-Step Results** (Intent + Function in unified call):
| Model | Intent Accuracy | Function Accuracy | Overall Accuracy | Avg Time |
|-------|-----------------|-------------------|------------------|----------|
| Llama-3.2-3B-Instruct | 80.1% | 80.1% | **80.1%** | 0.747s |
| gemma-2-2b-it | 79.5% | 79.2% | **79.2%** | 1.603s |
| Phi-3.5-mini-instruct | 56.9% | 53.5% | **53.5%** | 2.803s |

**Two-Step Results** (Sequential intent → function calls):  
| Model | Step1 Accuracy | Step2 Accuracy | Overall Accuracy | Avg Time |
|-------|----------------|----------------|------------------|----------|
| Phi-3.5-mini-instruct | 80.6% | 94.8% | **80.6%** | 3.376s |
| gemma-2-2b-it | 79.0% | 76.0% | **77.7%** | 2.615s |
| Llama-3.2-3B-Instruct | 72.2% | 76.1% | **71.6%** | 1.751s |

#### Architecture Performance Analysis

**Model-Specific Architectural Preferences**:

🔵 **Phi-3.5-mini-instruct**: **Strong Two-Step Preference (+27.1%)**
- Excels at function selection when given focused task (94.8% Step 2 accuracy)
- Struggles with complex unified prompts (53.5% single-step)
- Benefits significantly from task separation and specialized prompts

🟡 **Llama-3.2-3B-Instruct**: **Strong Single-Step Preference (+8.5%)**  
- Dominates unified reasoning tasks (80.1% single-step)
- Sequential two-step processing disrupts performance (71.6% two-step)
- Fastest inference across all approaches (0.747s single-step)

⚪ **gemma-2-2b-it**: **Architecture Agnostic (+1.5%)**
- Consistent performance across both approaches (79.2% vs 77.7%)
- Balanced capability for both unified and sequential processing
- Reliable baseline choice regardless of architecture

#### Updated Key Findings

**Best Overall Performance**:
- **Single-Step**: Llama-3.2-3B-Instruct (80.1% accuracy, 0.747s)
- **Two-Step**: Phi-3.5-mini-instruct (80.6% accuracy, 3.376s)

**Architecture Selection Guide**:
- **Choose Single-Step**: When speed is critical, using Llama-3.2-3B, or preferring simplicity
- **Choose Two-Step**: When using Phi-3.5-mini, need granular debugging, or preparing for MCP integration
- **Either Works**: gemma-2-2b-it performs consistently across both approaches

**Performance Validation**: These results represent the corrected evaluation system using consistent function names and proper dataset-driven evaluation. Previous results were affected by function name mismatches that have been resolved.

## Test Dataset Structure

Uses **BFCL (Berkeley Function Calling Leaderboard) format** with 157 test cases covering:

- **Intent Classification**: All test cases evaluate chat vs action classification
- **Function Calling**: Action cases evaluate function selection and parameter extraction
- **Real API Integration**: Uses actual backend functions (see [Function Reference](#function-reference))
- **Rich Parameter Extraction**: Natural language → structured function calls with parameters

**Example Test Case**:
```json
{
  "id": "action_001",
  "question": "find discussions about AI", 
  "intent_expected": "action",
  "expected_function_call": "search_threads(keyword_filter='AI')"
}
```

## How to run evaluations

### Setup Environment
```bash
# Create virtual environment
uv venv --python-preference only-managed

# Install MLC dependencies (choose one option)
# Option 1: Stable (CPU-only, recommended)
uv pip install --find-links https://mlc.ai/wheels mlc_llm_cpu==0.19.0 mlc_ai_cpu==0.19.0

# Option 2: Nightly (includes Metal GPU on Apple Silicon)  
uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly

# Install additional dependencies
uv pip install tqdm
```

### Download Models
```bash
# Install git-lfs
brew install git-lfs && git lfs install

# Download models to models/ directory
cd models
git clone https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC
git clone https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC
cd ..
```

### Run Evaluations
```bash
# Quick smoke test (4 basic queries)
uv run python mlc_llm/model_wrapper.py

# Run both evaluations with comparison (recommended)
uv run python mlc_llm/eval.py --iterations 10 --quiet

# Single-step evaluation only
uv run python mlc_llm/eval.py --mode single --iterations 10 --quiet

# Two-step evaluation only
uv run python mlc_llm/eval.py --mode two --iterations 10 --quiet

# Test with specific settings
uv run python mlc_llm/eval.py --temp 0.2 --iterations 5 --dataset pinned_thread_summary
```

### Results Storage

All evaluation results are automatically timestamped and stored in `mlc_llm/past_evals/` for historical tracking:

```
mlc_llm/past_evals/
├── single_step_2025-08-14_10-30-15.json     # Single-step evaluation results
├── two_step_2025-08-14_10-30-15.json        # Two-step evaluation results  
└── comparison_2025-08-14_10-30-15.json      # Comparative analysis (when both run)
```

This enables:
- **Historical Performance Tracking**: Compare model performance over time
- **Like-for-Like Comparisons**: Same timestamp ensures identical evaluation conditions
- **Regression Detection**: Identify when changes impact model performance
- **Result Archival**: All evaluation data preserved for future analysis

### Extending Evaluations

**Add New Test Cases**:
1. Edit `bfcl_testcases.json` with new test cases
2. Run unified evaluation: `uv run python mlc_llm/eval.py --iterations 10`
3. Compare results with previous evaluations in `past_evals/`

**Test New Models**:
1. Download model to `models/` directory
2. Run smoke test: `uv run python mlc_llm/model_wrapper.py --model <model-name>`
3. Run full evaluation: `uv run python mlc_llm/eval.py --iterations 10`
4. Check timestamped results in `past_evals/` for performance comparison

**Add New Functions**:
1. Define function schema in BFCL format
2. Add test cases with expected function calls
3. Update prompts in `src/prompts/` if needed
4. Run evaluation to verify new function handling

### Requirements
- **Unified Script**: Use `eval.py` for all evaluations (replaces separate single-step/two-step scripts)
- **Prompts**: Evaluation requires `src/prompts/singleStep.ts` and `src/prompts/actionOnly.ts` from the TypeScript source
- **Dataset**: Uses `bfcl_testcases.json` with 157 test cases
- **Models**: Download models to `models/` directory using git-lfs
- **Results**: Timestamped results automatically saved to `past_evals/` directory

## Future Enhancements

- **More simple actions and evaluations** using user feedback
- **Multi-step intent detection and reasoning** for complex queries
- **Contextual awareness** using conversation history
- **Evaluate function calling models** for finding higher accuracies
