# Past Evaluations

This directory contains timestamped evaluation results from the unified evaluation script (`eval.py`).

## File Naming Convention

- `single_step_YYYY-MM-DD_HH-MM-SS.json` - Single-step evaluation results
- `two_step_YYYY-MM-DD_HH-MM-SS.json` - Two-step evaluation results  
- `comparison_YYYY-MM-DD_HH-MM-SS.json` - Comparative analysis between approaches

## Usage

Results are automatically saved here when running:

```bash
# Run both evaluations and generate comparison
uv run python mlc_llm/eval.py

# Run specific evaluation mode
uv run python mlc_llm/eval.py --mode single
uv run python mlc_llm/eval.py --mode two
```

Each JSON file contains:
- Evaluation metadata (timestamp, parameters, test cases)
- Model performance metrics
- Detailed per-test-case analysis
- Aggregate statistics

This enables historical tracking and like-for-like comparisons across different evaluation runs.