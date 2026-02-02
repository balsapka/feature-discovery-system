# Feature Discovery System Overview

## Workflow Architecture

The system uses a **LangGraph-based iterative refinement pipeline** with an evaluator-optimizer pattern:

```
SUMMARIZE → GENERATE → EVALUATE → [CONVERGED?]
                                      ↓
                                NO → Loop back to GENERATE
                                YES → FINALIZE → END
```

**State** flows through nodes containing the business problem, current features, evaluations, iteration count, and convergence status.

---

## The Three Agents

| Agent | File | Purpose |
|-------|------|---------|
| **Summarizer** | `src/agents/summarizer.py` | Condenses business problems >500 chars; extracts USER-REQUESTED FEATURES section |
| **Generator** | `src/agents/feature_generator.py` | Creates candidate features with name, description, data_type, taxonomy, rationale |
| **Evaluator** | `src/agents/evaluator.py` | Creates problem-specific rubrics; scores features 0-10; decides convergence |

---

## Running Modes

### Standard Mode (default)

- Full feature schema with taxonomy and rationale
- Dynamic rubric generation (4-6 weighted criteria)
- Detailed per-criterion scoring

### Compact Mode (`compact_mode=True`)

**Purpose**: Faster execution with minimal schemas and tokens.

| Component | Standard | Compact |
|-----------|----------|---------|
| **Generator prompt** | ~800 tokens | ~200 tokens |
| **Feature schema** | name, description, data_type, taxonomy, rationale | name, desc (max 15 words), type |
| **Rubric** | LLM-generated 4-6 criteria | Fixed 3 criteria (Relevance 0.4, Feasibility 0.3, Predictive 0.3) |
| **Evaluation prompt** | Full feature details | Feature names only |
| **Output parser** | `JsonOutputParser` | `RobustJsonOutputParser` (handles malformed JSON) |

**Trade-off**: ~2x faster but less detailed feedback.

### Parallel Mode (`parallel=True`)

**Purpose**: ~2x speedup via concurrent LLM calls.

#### Generation (parallel)

- Splits into 2 focus areas: `"behavioral and transactional"` + `"demographic and external"`
- Uses `ThreadPoolExecutor` with `num_batches` workers
- Each batch generates `target_feature_count / num_batches` features
- Results deduplicated by feature name
- Only used for initial generation, not regeneration

**Code reference**: `feature_generator.py:195-200, 232-284`

#### Evaluation (parallel)

- Splits features into batches of `eval_batch_size` (default 10)
- Evaluates batches concurrently via `ThreadPoolExecutor`
- Failed batches automatically retried sequentially

**Code reference**: `evaluator.py:318-365`

---

## Key Guardrails for Correctness

### 1. Schema Validation (Pydantic)

- Scores constrained to `0-10` via `Field(ge=0, le=10)`
- Rubric weights constrained to `0-1`
- Data types enum-constrained: `STRUCTURED | UNSTRUCTURED | TIME_SERIES | EXTERNAL`

**Code reference**: `src/models/schemas.py`

### 2. Feature Name Normalization

All feature names are forced to `lowercase_snake_case` at generation time:

```python
def _normalize_feature_name(self, name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")
```

**Code reference**: `feature_generator.py:202-204`

### 3. Rejected Feature Tracking

- `rejected_feature_names` set prevents regenerating the same features across iterations
- Passed to LLM prompt AND filtered post-generation as a safety net

**Code reference**: `feature_generator.py:322-325, 412`

### 4. Score Carryover

- High-scoring features (≥7.0) are kept and their scores carried forward—not re-evaluated
- Only new/replacement features go through evaluation
- Reduces token usage and prevents score drift

**Code reference**: `evaluator.py:251-258`

### 5. Data Type Distribution Preservation

When regenerating low-scoring features, the system tracks `low_scoring_type_counts` and instructs the LLM to match the same distribution:

```python
type_instructions = f"{count} {dtype}" for each dtype in low_scoring_type_counts
```

**Code reference**: `evaluator.py:479-483`

### 6. User-Requested Features Protection

If the business problem contains a "USER-REQUESTED FEATURES" section, those features are **mandatory** and must appear first in the output.

```
**CRITICAL: If the business problem contains a "USER-REQUESTED FEATURES" section
or explicitly mentions specific features/data sources the user wants, you MUST
include those features FIRST in your output. These are non-negotiable requirements.**
```

**Code reference**: `feature_generator.py:64-66`

### 7. Evaluation Tracking (Audit Trail)

`EvaluationTracker` provides comprehensive accounting:

```python
class EvaluationTracker(BaseModel):
    total_input_features: int
    carried_features: List[str]       # Scored in previous iteration
    features_to_evaluate: List[str]   # New features to evaluate
    newly_scored: List[str]           # Successfully scored by LLM
    dropped_no_score: List[str]       # Dropped due to missing score
    kept_above_threshold: List[str]
    below_threshold: List[str]
```

**Code reference**: `evaluator.py:244-246`, `src/models/schemas.py`

### 8. Robust JSON Parsing (Compact Mode)

`RobustJsonOutputParser` handles common LLM output issues:

- Extracts JSON from markdown code blocks
- Fixes trailing commas
- Completes mismatched brackets/braces
- Handles truncated array elements

**Code reference**: `evaluator.py:41-94`

### 9. Convergence Safety

- `max_iterations` hard limit prevents infinite loops
- Convergence when no low-scoring features remain OR iterations exhausted

```python
should_continue = low_scoring_count > 0 and iteration < max_iterations - 1
```

**Code reference**: `workflow.py:296`, `evaluator.py:476`

### 10. Parallel Batch Failure Recovery

- Failed parallel batches are automatically retried sequentially
- All features in permanently failed batches are tracked as dropped

**Code reference**: `evaluator.py:349-363`

### 11. Empty Input Validation

- Raises `ValueError` for empty business problems
- Raises `EvaluationError` for empty feature lists or unparseable scores

**Code reference**: `workflow.py:328-329`, `evaluator.py:239-242, 276-277`

---

## Configuration Parameters

```python
FeatureDiscoveryWorkflow(
    llm_provider="anthropic",          # or "openai"
    model_name="claude-sonnet-4-5-20250929",
    max_iterations=3,                  # Safety bound for iterations
    score_threshold=7.0,               # Features below this get regenerated
    target_feature_count=20,           # Initial generation target
    compact_mode=False,                # Use compact evaluator/generator
    parallel=False,                    # Enable parallel generation/evaluation
    num_batches=2,                     # Parallel generation batches
    eval_batch_size=10,                # Features per evaluation LLM call
    temperature=0.7
)
```

### Parameter Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_provider` | `"anthropic"` | LLM provider: `"anthropic"` or `"openai"` |
| `model_name` | `"claude-sonnet-4-5-20250929"` | Specific model to use |
| `max_iterations` | `3` | Maximum generate-evaluate cycles |
| `score_threshold` | `7.0` | Minimum score to keep a feature |
| `target_feature_count` | `20` | Number of features to generate initially |
| `compact_mode` | `False` | Use minimal prompts/schemas for speed |
| `parallel` | `False` | Enable concurrent LLM calls |
| `num_batches` | `2` | Number of parallel generation batches |
| `eval_batch_size` | `10` | Features per evaluation batch |
| `temperature` | `0.7` | LLM temperature for generation |

---

## Data Flow Summary

```
BUSINESS PROBLEM (input)
        ↓
[SUMMARIZE NODE]
    - Condenses if >500 chars
    - Extracts user-requested features
        ↓
[GENERATE NODE - ITERATION 1]
    - Generates 20 features (full set)
    - Parallel mode: 2 concurrent batches by focus area
        ↓
[EVALUATE NODE - ITERATION 1]
    - Creates rubric (StandardEvaluator) or uses fixed rubric (CompactEvaluator)
    - Scores all features against rubric
    - Splits into: kept (≥7.0) + low-scoring (<7.0)
    - Calculates: avg_score, should_continue
        ↓
[CONDITIONAL: should_continue?]
    NO → FINALIZE
    YES ↓
        [GENERATE NODE - ITERATION 2+]
            - Takes: kept_features, rejected_names, type_distribution
            - Generates: only replacement features
            - Combines: kept + new
                ↓
        [EVALUATE NODE - ITERATION 2+]
            - Carries forward kept scores (no re-eval)
            - Evaluates new features only
            - Repeats split logic
                ↓
        [CONVERGED?]
            ...loops until converged or max_iterations reached
        ↓
[FINALIZE NODE]
    - Packages features, rubric, evaluations, stats
        ↓
FINAL OUTPUT (JSON)
```

---

## File Structure

```
feature-discovery-system/
├── src/
│   ├── agents/
│   │   ├── summarizer.py      # Business problem summarization
│   │   ├── feature_generator.py   # Feature generation (standard + compact + parallel)
│   │   └── evaluator.py       # Rubric creation & scoring (standard + compact + parallel)
│   ├── models/
│   │   └── schemas.py         # Pydantic models with validation
│   ├── workflow.py            # LangGraph orchestration
│   └── cli.py                 # Command-line interface
└── tests/
    └── test_workflow.py       # Unit tests
```

---

## Usage Example

```python
from src.workflow import FeatureDiscoveryWorkflow

# Standard mode
workflow = FeatureDiscoveryWorkflow(
    llm_provider="anthropic",
    max_iterations=3,
    score_threshold=7.0
)

# Fast mode (compact + parallel)
workflow = FeatureDiscoveryWorkflow(
    llm_provider="anthropic",
    compact_mode=True,
    parallel=True,
    max_iterations=2
)

# Run
result = workflow.run(
    business_problem="Predict customer churn for a retail bank...",
    verbose=True
)

# Access results
features = result["features"]
evaluations = result["evaluations"]
total_iterations = result["total_iterations"]
```

---

*Generated: 2026-02-02*
