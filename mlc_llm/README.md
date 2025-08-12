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
  - [Planned Improvements](#planned-improvements)
  - [MCP Integration Readiness](#mcp-integration-readiness)

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

### `get_thread_cards`
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

### Single-Step vs Two-Step Comparison (144/158 Test Cases, 10 Iterations)

> **Note**: Single-step evaluation uses 144 test cases, two-step uses 158 test cases (slightly different dataset versions), but both use 10 iterations for statistical reliability.

#### Single-Step Results
| Model | Overall Accuracy | Avg Time |
|-------|------------------|----------|
| Llama-3.2-3B-Instruct | 81.1% | 0.829s |
| gemma-2-2b-it | 76.7% | 1.695s |
| Phi-3.5-mini-instruct | 72.3% | 2.953s |

#### Two-Step Results  
| Model | Overall Accuracy | Total Time |
|-------|------------------|------------|
| Phi-3.5-mini-instruct | 77.3% | 4.890s |
| gemma-2-2b-it | 73.4% | 2.841s |
| Llama-3.2-3B-Instruct | 70.8% | 2.467s |

#### Performance Differences

**Llama-3.2-3B-Instruct**:
- Single-step: 81.1% vs Two-step: 70.8% = +10.3% accuracy gain
- Speed: 0.829s vs 2.467s = 3.0x faster

**gemma-2-2b-it**:
- Single-step: 76.7% vs Two-step: 73.4% = +3.3% accuracy gain  
- Speed: 1.695s vs 2.841s = 1.7x faster

**Phi-3.5-mini-instruct**:
- Single-step: 72.3% vs Two-step: 77.3% = -5.0% accuracy loss
- Speed: 2.953s vs 4.890s = 1.7x faster

#### Overall Summary
- Average accuracy: Single-step 76.7% vs Two-step 73.8% = +2.9% difference
- Speed improvement: 2x-3x faster inference across all models
- Architecture simplicity: One prompt vs two sequential LLM calls
- Resource efficiency: Lower memory usage, fewer model loads

## Test Dataset Structure

Uses **BFCL (Berkeley Function Calling Leaderboard) format** with 158 test cases covering:

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
  "expected_function_call": "get_thread_cards(keyword_filter='AI')"
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

# Single-step evaluation (recommended)
uv run python mlc_llm/eval_single_step.py --iterations 10 --quiet

# Two-step evaluation (for comparison)
uv run python mlc_llm/eval_two_step.py --iterations 10 --quiet

# Test specific model
uv run python mlc_llm/eval_single_step.py --model "Phi-3.5-mini-instruct-q4f16_1-MLC"
```

### Extending Evaluations

**Add New Test Cases**:
1. Edit `bfcl_testcases.json` with new test cases
2. Run evaluation to establish baseline

**Test New Models**:
1. Download model to `models/` directory
2. Run smoke test: `uv run python mlc_llm/model_wrapper.py --model <model-name>`
3. Run full evaluation and compare results

**Add New Functions**:
1. Define function schema in BFCL format
2. Add test cases with expected function calls
3. Update evaluation scripts for new functionality

### Requirements
- **Prompts**: Evaluation scripts require `src/prompts/singleStep.ts` and `src/prompts/actionOnly.ts` from the TypeScript source
- **Dataset**: Uses `bfcl_testcases.json` with 158 test cases
- **Models**: Download models to `models/` directory using git-lfs

## Future Enhancements

### Planned Improvements
- **Multi-step intent detection** for complex queries
- **Contextual awareness** using conversation history

### MCP Integration Readiness
- **Multi-step reasoning** capabilities for complex queries
- **Function calling patterns** prepared for MCP server integration
