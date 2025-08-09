# Intent Detection Model Evaluation

This directory contains tools for evaluating and testing intent detection models used in the Techne browser extension.

## Overview

The intent detection system determines whether user messages are asking to **SEARCH** for existing discussions or asking for **CONVERSATIONAL** responses. This is critical for routing user requests appropriately in the chat interface.

## Scripts

### `quick-intent-test.py`
Interactive testing tool for individual queries or batch testing.

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

### `eval_intent_detection.py`
Comprehensive evaluation framework with advanced metrics and model comparison.

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

### Tested Models Performance

| Model | Size | Accuracy | Precision | Recall | F1 Score | Production Ready |
|-------|------|----------|-----------|--------|----------|------------------|
| **Phi-3.5-mini-instruct** | 2.2GB | **85.9%** | **0.86** | **0.86** | **0.86** | ✅ **RECOMMENDED** |
| Gemma-2-2B-it | 1.4GB | 81.2% | 0.81 | 0.81 | 0.81 | ✅ Resource-constrained |
| Llama-3.2-3B-Instruct | 1.9GB | 82.3% | 0.82 | 0.82 | 0.82 | ✅ Good fallback |
| DeepSeek-R1-Distill-Qwen-7B | 4.2GB | 73.4% | 0.73 | 0.73 | 0.73 | ⚠️ Needs fixes |

### Key Findings

#### ✅ Phi-3.5-mini-instruct (RECOMMENDED)
- **Highest accuracy** at 85.9% with excellent JSON reliability
- **Best balance** of performance, size (2.2GB), and reliability
- **Strong disambiguation** between search and conversational intents
- **Production ready** with robust error handling

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

The evaluation uses 64+ carefully crafted test cases across multiple categories:

### Explicit Search Intent (24 cases)
- Direct search commands: "find discussions about AI"
- Show/lookup requests: "show me posts about React"  
- Question format searches: "any discussions on blockchain?"

### Conversational Intent (21 cases)
- Explanation requests: "what do you think about AI?"
- Definition questions: "what is machine learning?"
- Opinion seeking: "how do you feel about startups?"

### Ambiguous Cases (10 cases)  
- Context-dependent: "what about React?"
- Unclear intent: "tell me more"
- Borderline cases requiring inference

### Edge Cases (9 cases)
- Social interactions: "hello", "thanks", "good morning"
- Empty/minimal input: "", "hmm", "ok"
- Complex multi-part queries

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
python mlc_llm/eval_intent_detection.py

# Test edge cases specifically  
python mlc_llm/eval_intent_detection.py --dataset edge_cases

# Verify production model performance
python mlc_llm/eval_intent_detection.py --model "Phi-3.5-mini-instruct-q4f16_1-MLC"
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