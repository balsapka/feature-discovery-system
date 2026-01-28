# Feature Discovery System

A multi-agent system for data scientists in banking to discover relevant features from an internal Feature Store based on business problems.

## Overview

This system uses LangGraph to orchestrate multiple agents that:
1. Generate candidate features from natural language business problems
2. Create evaluation rubrics and iteratively refine features
3. Search the internal Feature Store using semantic similarity
4. Rank and return the top-k most relevant features

## Architecture

### Phase 1: Feature Generation & Evaluation (Current)
- **Feature Generator Agent**: Generates candidate features with taxonomy
- **Evaluator Agent**: Creates problem-specific rubrics and scores features
- **Iterative Refinement**: Generator and Evaluator loop until convergence

### Phase 2: Feature Store Integration (Future)
- Vector search in Qdrant Feature Store
- Ranking agent for final feature selection

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```bash
# Choose your LLM provider
OPENAI_API_KEY=your_key_here
# OR
ANTHROPIC_API_KEY=your_key_here

# Model configuration
LLM_PROVIDER=anthropic  # or 'openai'
MODEL_NAME=claude-sonnet-4-5-20250929  # or 'gpt-4' etc.

# System configuration
MAX_ITERATIONS=3
```

## Usage

```python
from src.workflow import FeatureDiscoveryWorkflow

# Initialize the workflow
workflow = FeatureDiscoveryWorkflow(
    llm_provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    max_iterations=3
)

# Run feature discovery
business_problem = """
Predict customer churn in retail banking. We want to identify customers 
likely to close their accounts in the next 90 days.
"""

result = workflow.run(business_problem)

print("Generated Features:", result['features'])
print("Evaluation Rubric:", result['rubric'])
print("Final Scores:", result['scores'])
```

## Project Structure

```
feature-discovery-system/
├── src/
│   ├── agents/
│   │   ├── feature_generator.py    # Feature generation agent
│   │   ├── evaluator.py            # Evaluation & rubric creation
│   │   └── __init__.py
│   ├── models/
│   │   └── schemas.py              # Pydantic models
│   ├── workflow.py                 # LangGraph workflow orchestration
│   └── __init__.py
├── tests/
│   └── test_workflow.py
├── examples/
│   └── basic_example.py
├── requirements.txt
├── .env.example
└── README.md
```

## Development Status

- [x] Phase 1: Feature Generator & Evaluator agents
- [ ] Phase 2: Qdrant Feature Store integration
- [ ] Phase 3: Ranking agent
- [ ] Phase 4: Advanced features (aggregations, transformations)

## License

MIT
