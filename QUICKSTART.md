# Feature Discovery System - Quick Reference

## Project Structure

```
feature-discovery-system/
├── src/
│   ├── agents/
│   │   ├── feature_generator.py    # Generates candidate features
│   │   ├── evaluator.py            # Creates rubrics & evaluates
│   │   └── __init__.py
│   ├── models/
│   │   ├── schemas.py              # Pydantic data models
│   │   └── __init__.py
│   ├── workflow.py                 # LangGraph orchestration
│   └── __init__.py
├── tests/
│   └── test_workflow.py            # Unit tests
├── examples/
│   └── basic_example.py            # Usage example
└── requirements.txt
```

## Setup Instructions

1. **Install dependencies:**
   ```bash
   ./setup.sh
   # OR manually:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

3. **Run example:**
   ```bash
   python examples/basic_example.py
   ```

## Architecture Overview

### LangGraph Workflow

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  GENERATE       │ ← Feature Generator Agent
│  (features)     │   - Generates 8-15 candidate features
└──────┬──────────┘   - Creates taxonomy
       │
       ▼
┌─────────────────┐
│  EVALUATE       │ ← Evaluator Agent
│  (rubric+score) │   - Creates problem-specific rubric
└──────┬──────────┘   - Scores each feature
       │              - Suggests improvements
       ▼
    Decision:
    Continue? ────No──→ FINALIZE ──→ END
       │
      Yes (iterate)
       │
       └──────────────→ GENERATE (with feedback)
```

### Key Components

**1. Feature Generator Agent (`feature_generator.py`)**
- Generates features based on business problem
- Considers multiple data types (structured, time-series, unstructured, external)
- Creates taxonomy for organization
- Incorporates feedback from evaluator

**2. Evaluator Agent (`evaluator.py`)**
- Creates custom rubric for the problem
- Scores features on multiple criteria (relevance, availability, predictive power, etc.)
- Provides feedback for improvement
- Determines if iteration would help

**3. Workflow Orchestrator (`workflow.py`)**
- Uses LangGraph StateGraph
- Manages iteration loop
- Caps at max_iterations
- Finalizes and returns results

## Data Models

**Feature:**
```python
{
    "name": str,
    "description": str,
    "data_type": "structured|time_series|unstructured|external",
    "taxonomy": str,
    "rationale": str
}
```

**Evaluation Rubric:**
```python
{
    "criteria": [
        {
            "name": str,
            "description": str,
            "weight": float  # 0-1, sum to 1.0
        }
    ],
    "rationale": str
}
```

**Feature Score:**
```python
{
    "feature_name": str,
    "criterion_scores": {criterion: score},  # scores 0-10
    "overall_score": float,
    "feedback": str
}
```

## Usage Examples

### Basic Usage

```python
from src.workflow import FeatureDiscoveryWorkflow

workflow = FeatureDiscoveryWorkflow(
    llm_provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    max_iterations=3
)

result = workflow.run(
    business_problem="Predict customer churn in retail banking"
)

# Access results
features = result['features']
rubric = result['rubric']
scores = result['evaluations']
```

### With Custom Parameters

```python
workflow = FeatureDiscoveryWorkflow(
    llm_provider="openai",
    model_name="gpt-4-turbo-preview",
    max_iterations=5,
    temperature=0.8,
    api_key="your-key-here"
)

result = workflow.run(business_problem, verbose=True)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_workflow.py::TestFeatureDiscoveryWorkflow
```

## Configuration Options

**LLM Providers:**
- `anthropic`: Claude models (recommended)
- `openai`: GPT models

**Model Names:**
- Anthropic: `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`
- OpenAI: `gpt-4`, `gpt-4-turbo-preview`

**Parameters:**
- `max_iterations`: Number of generator-evaluator loops (default: 3)
- `temperature`: LLM creativity (0.0-1.0, default: 0.7)
- `verbose`: Print intermediate steps (default: False)

## Future Enhancements

**Phase 2: Feature Store Integration**
- Qdrant vector search
- Semantic similarity matching
- Ranking agent for final selection

**Phase 3: Advanced Features**
- Aggregated/transformed features
- Custom taxonomy templates
- Feature lineage tracking

## Troubleshooting

**Issue: API key not found**
```bash
# Check .env file exists and contains:
ANTHROPIC_API_KEY=sk-ant-...
# OR
OPENAI_API_KEY=sk-...
```

**Issue: Import errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue: LangGraph errors**
```bash
# Update LangGraph
pip install --upgrade langgraph langchain
```

## Contact & Feedback

For questions or issues, please refer to the main README.md or create an issue in the repository.
