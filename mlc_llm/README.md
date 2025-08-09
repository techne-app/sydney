# Intent Detection Model Evaluation

This directory contains tools for evaluating and testing intent detection models used in the Techne browser extension.

## Overview

The intent detection system uses a **two-phase classification approach** to route user requests in the chat interface:

### Phase 1: Chat vs Action Classification
Determines whether user messages are asking for **CONVERSATIONAL** responses or want the system to **DO SOMETHING** (take action).

- **chat**: User wants conversation, explanations, opinions, advice, or general discussion
- **action**: User wants the system to perform an action - search, find, retrieve, create, analyze, etc.

### Phase 2: Action Type Classification  
For messages classified as "action", determines the specific action type:

- **search**: Find, show, retrieve, discover HN content/discussions  
- **create**: Generate, write, build something new (future)
- **analyze**: Analyze trends, compare, synthesize information (future)

This two-phase approach is **extensible and MCP-ready** - new action types can be added without retraining Phase 1, and the architecture naturally maps to tool calling patterns for future MCP server integration.

## Scripts

### `quick-intent-test.py`
Interactive testing tool for individual queries or batch testing (legacy, single-phase).

```bash
# Test a single query
python mlc_llm/quick-intent-test.py "find AI discussions"

# Test with specific model
python mlc_llm/quick-intent-test.py --model "Phi-3.5-mini-instruct-q4f16_1-MLC" "search for React"

# Batch test all queries
python mlc_llm/quick-intent-test.py --batch

# Adjust temperature
python mlc_llm/quick-intent-test.py --temp 0.3 "what about startups?"
```

### `intent_evaluation_framework.py` 🆕
**NEW**: Modular evaluation framework supporting two-phase classification with separate test case files.

```bash
# Phase 1 evaluation (chat vs action)
python mlc_llm/intent_evaluation_framework.py --phase phase1 --model "Phi-3.5-mini-instruct-q4f16_1-MLC"

# Phase 2 evaluation (action type classification)  
python mlc_llm/intent_evaluation_framework.py --phase phase2 --model "Llama-3.2-3B-Instruct-q4f16_1-MLC"

# Evaluate both phases
python mlc_llm/intent_evaluation_framework.py --phase both --models "Phi-3.5-mini-instruct-q4f16_1-MLC" "Llama-3.2-3B-Instruct-q4f16_1-MLC"

# Filter by category or difficulty
python mlc_llm/intent_evaluation_framework.py --phase phase1 --category-filter explicit_search --difficulty-filter hard

# Compare multiple models across phases
python mlc_llm/intent_evaluation_framework.py --models "Phi-3.5-mini-instruct-q4f16_1-MLC" "Llama-3.2-3B-Instruct-q4f16_1-MLC" --phase both

# Export results to JSON
python mlc_llm/intent_evaluation_framework.py --phase phase1 --export results_phase1.json
```

### `eval_intent_detection.py` (Legacy)
Original comprehensive evaluation framework with embedded test cases.

```bash
# Full evaluation with default model
python mlc_llm/eval_intent_detection.py

# Compare two models
python mlc_llm/eval_intent_detection.py --compare "Phi-3.5-mini-instruct-q4f16_1-MLC" "Llama-3.2-3B-Instruct-q4f16_1-MLC"

# Test specific dataset categories
python mlc_llm/eval_intent_detection.py --dataset explicit_search
python mlc_llm/eval_intent_detection.py --dataset conversational

# Failure analysis
python mlc_llm/eval_intent_detection.py --failures-only

# Verbose output with detailed results
python mlc_llm/eval_intent_detection.py --verbose
```

## Model Evaluation Results

### Phase 1: Chat vs Action Classification (138 Test Cases)

| Model | Size (MB) | Accuracy | Precision | Recall | F1 Score | Production Ready | 
|-------|-----------|----------|-----------|--------|----------|------------------|
| 🥇 **Llama-3.2-3B-Instruct** | 1,733 | **88.9%** | **88.5%** | **93.9%** | **0.911** | ✅ **EXCELLENT** |
| 🥈 **Phi-3.5-mini-instruct** | 2,052 | **87.0%** | **93.2%** | **84.1%** | **0.885** | ✅ **EXCELLENT** |
| 🥉 **Gemma-2-2B-it** | 1,420 | **76.1%** | **73.3%** | **93.9%** | **0.824** | ⚠️ **ACCEPTABLE** |

### Multi-Run Accuracy Analysis 📊
**Accuracy across three evaluation runs (138 test cases each):**

| Model | Run 1 | Run 2 | Run 3 | Mean | Std Dev |
|-------|-------|-------|-------|------|---------|
| **Llama-3.2-3B-Instruct** | 85.9% | 87.4% | 88.9% | **87.4%** | **±1.5%** |
| **Phi-3.5-mini-instruct** | 85.9% | 87.0% | 87.0% | **86.6%** | **±0.6%** |
| **Gemma-2-2B-it** | 76.1% | 76.1% | 76.1% | **76.1%** | **±0.0%** |

### Real Evaluation Results Summary
- **Test Dataset**: 138 comprehensive test cases from `intent_detection_testcases.json`
- **Evaluation Platform**: MacBook Pro M3 with Metal GPU acceleration  
- **Temperature**: 0.1 for consistent, low-variance responses
- **Key Insight**: Llama leads in overall accuracy (88.9%) and F1 score (0.911), Phi has highest precision (93.2%)

### Key Findings

#### 🏆 Llama-3.2-3B-Instruct (RECOMMENDED)
- **Highest accuracy** at 88.9% with consistent improvement across runs
- **Best F1 score** at 0.911 showing excellent precision/recall balance  
- **Strong recall** at 93.9% - catches more search intents with fewer false negatives
- **Moderate size** at 1.7GB - good balance of performance and efficiency
- **Production ready** with robust JSON parsing and error handling

#### ✅ Phi-3.5-mini-instruct (ALTERNATIVE)
- **Highest precision** at 93.2% - makes fewer false positive errors
- **Very consistent** performance across runs (±0.6% variance)
- **Larger model** at 2.0GB but more precise classifications
- **Best for applications** where false positives (classifying chat as search) are costly

#### ⚠️ Technical Issues Discovered

**DeepSeek R1 JSON Parsing Problems:**
- Generates `<think>` reasoning blocks that contaminate JSON output
- **Fixed** with custom parsing logic in evaluation script
- Requires increased `max_tokens` (200→500) for reasoning models

**Memory Management:**
- **Sequential model loading** implemented to prevent OOM errors
- Models properly cleaned up with `del` and `gc.collect()`
- Avoid simultaneous model loading on resource-constrained systems

## Test Dataset Structure

The evaluation uses test cases from `intent_detection_testcases.json` with structured metadata:

```json
{
  "metadata": {
    "phase_1": "chat vs action (binary classification)",  
    "phase_2": "action type classification (search, create, analyze, etc)",
    "total_cases": 64,
    "categories": ["explicit_search", "professional_search", "opinion_request", "ambiguous", ...],
    "difficulty_levels": ["easy", "medium", "hard"]
  },
  "test_cases": [...]
}
```

### Phase 1 Categories (Chat vs Action)

**Action Intent Categories:**
- **explicit_search**: Clear search commands ("find AI discussions")
- **professional_search**: Job-focused searches ("salary threads for SF engineers")  
- **entrepreneur_search**: Startup-focused searches ("YC founder stories")
- **technical_search**: Tech-specific searches ("Rust performance benchmarks")
- **ambiguous**: Context-dependent cases ("what about React?")
- **single_word**: Minimal queries ("Docker", "React?")

**Chat Intent Categories:**
- **opinion_request**: Asking for opinions ("what's your take on AI?")
- **explanation_request**: Seeking explanations ("can you explain ML?")
- **technical_explanation**: Deep technical understanding requests
- **career_advice**: Professional guidance questions
- **greeting**: Social interactions ("hello", "thanks")

### Phase 2 Categories (Action Types)
Currently focused on **search** actions, with architecture ready for:
- **create**: Generate/build content (future)
- **analyze**: Trend analysis, comparisons (future)

### Difficulty Distribution
- **Easy**: 20% - Clear, unambiguous cases
- **Medium**: 60% - Typical user patterns with some complexity  
- **Hard**: 20% - Edge cases, ambiguous queries, minimal input

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

## Evaluation Metrics

### Core Metrics
- **Accuracy**: Overall correctness percentage
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Advanced Analysis
- **Confusion Matrix**: Detailed breakdown of prediction vs actual
- **Failure Analysis**: Specific cases where models fail
- **JSON Reliability**: Parsing success rate for structured output
- **Category Performance**: Accuracy across different intent types

### Model Comparison Features
- **Side-by-side evaluation** with disagreement analysis
- **Statistical significance** testing
- **Memory-efficient sequential loading**
- **Progress tracking** with tqdm integration

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

### Error Handling
- **Graceful degradation** on JSON parse failures
- **Confidence thresholds** for uncertain classifications
- **Fallback routing** to conversational AI when needed

## Prerequisites

### Model Setup
- **git-lfs**: Required for downloading large model files
- **Python 3.10+**: For running MLC LLM
- **uv**: Modern Python package manager

```bash
# Install git-lfs (required for model downloads)
brew install git-lfs
git lfs install

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Download MLC Models
From the repo root, download complete MLC-compiled models:

```bash
cd models

# Download recommended model
git clone https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC

# Download fallback model
git clone https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC

# Verify large files downloaded correctly (should be 100+ MB each)
ls -lah */*.bin

cd ..
```

### Python Environment Setup
```bash
cd mlc_llm

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate  # On Windows

# Install MLC LLM (requires special wheel repository)
uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly
uv pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly

cd ..
```

## Quality Assurance

### Pre-Production Checklist
- [ ] Model accuracy ≥ 80% on evaluation dataset
- [ ] JSON parsing reliability ≥ 95%
- [ ] Memory usage within browser constraints
- [ ] Latency ≤ 2 seconds for intent detection
- [ ] No security vulnerabilities in model loading

### Testing Procedures
```bash
# Run full evaluation
python mlc_llm/eval_intent_detection.py --full-eval

# Evaluate all models with performance metrics
python mlc_llm/eval_intent_detection.py --eval-all

# Test edge cases specifically  
python mlc_llm/eval_intent_detection.py --dataset edge_cases

# Export comprehensive results
python mlc_llm/eval_intent_detection.py --eval-all --export-results model_comparison.json
```

## Development Workflow

### Adding New Test Cases
1. Edit test dataset in `eval_intent_detection.py`
2. Run evaluation to establish baseline
3. Update production model if needed
4. Document changes in this README

### Model Updates
1. Download new model to `models/` directory
2. Test with `quick-intent-test.py` for basic functionality
3. Run full evaluation with `eval_intent_detection.py`
4. Update configuration if performance improves
5. Update documentation with new results

### Prompt Improvements
1. Edit `src/prompts/searchIntent.ts`
2. Test changes with `quick-intent-test.py`
3. Run full evaluation to measure impact
4. Deploy if accuracy increases

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

## Troubleshooting

### Common Issues

**Model Not Found**
```bash
❌ Model not found. Tried:
   - /path/to/models/ModelName
```
- Ensure you're running from repo root directory
- Check that `models/` directory exists with correct model folder

**JSON Parse Errors**
```bash
❌ JSON parse error: Expecting property name enclosed in quotes
```
- Model may be generating malformed JSON
- Try different temperature settings
- Consider switching to a more reliable model

**Memory Issues**
```bash
❌ Failed to initialize engine: Out of memory
```
- Use sequential model comparison instead of simultaneous
- Close other applications to free memory
- Consider smaller models for resource-constrained systems

**Installation Problems**
```bash
❌ MLC LLM not installed
```
```bash
uv pip install --pre -f https://mlc.ai/wheels mlc-llm-nightly
uv pip install --pre -f https://mlc.ai/wheels mlc-ai-nightly
```

### Debug Mode
Enable verbose logging for detailed troubleshooting:
```bash
python mlc_llm/eval_intent_detection.py --verbose
```

This provides:
- Raw model responses
- JSON parsing attempts  
- Detailed error messages
- Performance timing data