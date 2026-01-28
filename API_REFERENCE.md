# API Reference

## Core Classes

### FeatureDiscoveryWorkflow

Main orchestrator for the feature discovery process.

```python
from src.workflow import FeatureDiscoveryWorkflow

workflow = FeatureDiscoveryWorkflow(
    llm_provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    max_iterations=3,
    temperature=0.7,
    api_key=None  # Optional, loads from .env if not provided
)
```

#### Parameters

- **llm_provider** (str): LLM provider - 'anthropic' or 'openai'
- **model_name** (str, optional): Specific model name. Defaults based on provider
- **max_iterations** (int): Maximum number of generator-evaluator iterations. Default: 3
- **temperature** (float): LLM temperature for generation (0.0-1.0). Default: 0.7
- **api_key** (str, optional): API key. If None, loads from environment

#### Methods

##### run(business_problem, verbose=False)

Run the complete feature discovery workflow.

```python
result = workflow.run(
    business_problem="Predict customer churn in retail banking",
    verbose=True
)
```

**Parameters:**
- **business_problem** (str): Natural language description of the business problem
- **verbose** (bool): If True, prints intermediate states. Default: False

**Returns:** Dictionary with keys:
- **features** (List[dict]): Generated features
- **rubric** (dict): Evaluation rubric
- **evaluations** (List[dict]): Feature scores
- **taxonomy_explanation** (str): Feature organization explanation
- **total_iterations** (int): Number of iterations performed

**Example:**
```python
{
  "features": [...],
  "rubric": {...},
  "evaluations": [...],
  "taxonomy_explanation": "Features organized by...",
  "total_iterations": 2
}
```

##### visualize(output_path="workflow_graph.png")

Generate a visualization of the workflow graph.

```python
workflow.visualize("my_workflow.png")
```

---

## Agent Classes

### FeatureGeneratorAgent

Generates candidate features from business problems.

```python
from src.agents import FeatureGeneratorAgent

generator = FeatureGeneratorAgent(llm)
```

#### Methods

##### generate(business_problem, feedback="")

Generate features for a business problem.

```python
output = generator.generate(
    business_problem="Predict loan defaults",
    feedback="Add more behavioral features"
)
```

**Parameters:**
- **business_problem** (str): Natural language problem description
- **feedback** (str): Optional feedback from previous evaluation

**Returns:** `GeneratorOutput` object with:
- **features** (List[Feature]): Generated features
- **taxonomy_explanation** (str): Taxonomy structure explanation

---

### EvaluatorAgent

Creates rubrics and evaluates features.

```python
from src.agents import EvaluatorAgent

evaluator = EvaluatorAgent(llm)
```

#### Methods

##### create_rubric(business_problem)

Create an evaluation rubric for the problem.

```python
rubric = evaluator.create_rubric("Predict customer churn")
```

**Parameters:**
- **business_problem** (str): Problem description

**Returns:** `EvaluationRubric` object

##### evaluate_features(business_problem, features, rubric, iteration, max_iterations)

Evaluate features against the rubric.

```python
evaluation = evaluator.evaluate_features(
    business_problem="Predict churn",
    features=feature_list,
    rubric=rubric,
    iteration=1,
    max_iterations=3
)
```

**Parameters:**
- **business_problem** (str): Problem context
- **features** (List[Feature]): Features to evaluate
- **rubric** (EvaluationRubric): Evaluation criteria
- **iteration** (int): Current iteration number
- **max_iterations** (int): Maximum allowed iterations

**Returns:** `FeatureEvaluation` object with:
- **rubric** (EvaluationRubric): The rubric used
- **feature_scores** (List[FeatureScore]): Scores for each feature
- **improvement_suggested** (bool): Whether to iterate again
- **improvement_recommendations** (str): Feedback for next iteration

---

## Data Models

### Feature

Represents a candidate feature.

```python
from src.models import Feature, DataType

feature = Feature(
    name="account_age_days",
    description="Days since account opening",
    data_type=DataType.STRUCTURED,
    taxonomy="customer/account_info",
    rationale="Newer accounts may churn more"
)
```

**Fields:**
- **name** (str): Feature name
- **description** (str): Detailed description
- **data_type** (DataType): Type of data source
- **taxonomy** (str): Category/classification
- **rationale** (str): Why this feature is relevant

---

### DataType

Enumeration of data source types.

```python
from src.models import DataType

DataType.STRUCTURED      # Transactional, demographic data
DataType.UNSTRUCTURED    # Text, documents
DataType.TIME_SERIES     # Historical patterns
DataType.EXTERNAL        # Macroeconomic, market data
```

---

### EvaluationRubric

Evaluation criteria for features.

```python
from src.models import EvaluationRubric, RubricCriterion

rubric = EvaluationRubric(
    criteria=[
        RubricCriterion(
            name="Relevance",
            description="How relevant to the problem",
            weight=0.4
        ),
        # ... more criteria
    ],
    rationale="Balanced rubric for churn prediction"
)
```

**Fields:**
- **criteria** (List[RubricCriterion]): Evaluation criteria
- **rationale** (str): Why this rubric is appropriate

---

### RubricCriterion

A single evaluation criterion.

**Fields:**
- **name** (str): Criterion name
- **description** (str): What this criterion evaluates
- **weight** (float): Importance weight (0-1, sum to 1.0)

---

### FeatureScore

Score for a single feature.

```python
from src.models import FeatureScore

score = FeatureScore(
    feature_name="account_age_days",
    criterion_scores={
        "Relevance": 8.5,
        "Data Availability": 9.0,
        "Predictive Power": 7.5
    },
    overall_score=8.3,
    feedback="Strong feature with good availability"
)
```

**Fields:**
- **feature_name** (str): Name of the feature
- **criterion_scores** (Dict[str, float]): Scores per criterion (0-10)
- **overall_score** (float): Weighted overall score
- **feedback** (str): Qualitative feedback

---

### FeatureEvaluation

Complete evaluation of all features.

**Fields:**
- **rubric** (EvaluationRubric): The rubric used
- **feature_scores** (List[FeatureScore]): Scores for all features
- **improvement_suggested** (bool): Whether to iterate
- **improvement_recommendations** (str): Feedback for next iteration

---

### GeneratorOutput

Output from the Feature Generator.

**Fields:**
- **features** (List[Feature]): Generated features
- **taxonomy_explanation** (str): How features are organized

---

### WorkflowState

State that flows through the LangGraph workflow.

**Fields:**
- **business_problem** (str): Input problem
- **current_features** (List[Feature]): Current feature set
- **rubric** (EvaluationRubric): Evaluation criteria
- **evaluations** (List[FeatureScore]): Current scores
- **iteration** (int): Current iteration number
- **max_iterations** (int): Maximum iterations
- **converged** (bool): Whether workflow has converged
- **improvement_recommendations** (str): Feedback for next round
- **taxonomy_explanation** (str): Taxonomy structure
- **final_output** (dict): Final packaged results

---

## CLI Reference

### Command Line Interface

```bash
python cli.py [options]
```

#### Options

**--problem** TEXT
- Business problem description
- If not provided, prompts interactively

**--provider** {anthropic|openai}
- LLM provider (default: anthropic)

**--model** TEXT
- Model name (default: provider-specific)

**--iterations** INTEGER
- Maximum iterations (default: 3)

**--temperature** FLOAT
- LLM temperature (default: 0.7)

**--output** PATH
- Output file path (JSON format)

**--verbose**
- Enable verbose output

**--no-banner**
- Suppress welcome banner

#### Examples

```bash
# Interactive mode
python cli.py

# Direct execution
python cli.py --problem "Predict loan defaults"

# With all options
python cli.py \
  --problem "Detect fraud in transactions" \
  --provider anthropic \
  --model claude-sonnet-4-5-20250929 \
  --iterations 5 \
  --temperature 0.8 \
  --output results.json \
  --verbose

# Using OpenAI
python cli.py \
  --provider openai \
  --model gpt-4-turbo-preview \
  --problem "Credit risk scoring"
```

---

## Environment Variables

### Required

**ANTHROPIC_API_KEY** or **OPENAI_API_KEY**
- Your LLM provider API key
- Set in `.env` file or environment

### Optional

**LLM_PROVIDER**
- Default provider ('anthropic' or 'openai')
- Can be overridden in code

**MODEL_NAME**
- Default model name
- Can be overridden in code

**MAX_ITERATIONS**
- Default maximum iterations
- Can be overridden in code

**TEMPERATURE**
- Default LLM temperature
- Can be overridden in code

### Example .env

```bash
ANTHROPIC_API_KEY=sk-ant-api03-...
LLM_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-4-5-20250929
MAX_ITERATIONS=3
TEMPERATURE=0.7
```

---

## Error Handling

### Common Exceptions

**ValueError**
- Raised when invalid LLM provider specified
- Raised for invalid configuration parameters

**OutputParserException** (from LangChain)
- Raised when LLM output doesn't match expected schema
- Usually indicates prompt engineering issue

**APIError** (from LLM providers)
- Raised for API connection issues
- Check API key and network connectivity

### Example Error Handling

```python
from langchain.output_parsers.json import OutputParserException

try:
    workflow = FeatureDiscoveryWorkflow(llm_provider="anthropic")
    result = workflow.run(business_problem)
except ValueError as e:
    print(f"Configuration error: {e}")
except OutputParserException as e:
    print(f"LLM output parsing failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Type Hints

All classes and methods include full type hints for IDE support:

```python
from typing import List, Dict, Optional, Literal

def run(
    self, 
    business_problem: str, 
    verbose: bool = False
) -> Dict[str, Any]:
    ...
```

---

## Logging

### Enable Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now workflow will log progress
workflow = FeatureDiscoveryWorkflow()
result = workflow.run(business_problem)
```

### Log Levels

- **INFO**: Workflow progress, iterations
- **DEBUG**: Detailed execution info
- **WARNING**: Non-fatal issues
- **ERROR**: Failures and exceptions

---

## Performance Tips

### Optimize for Speed

```python
# Use fewer iterations
workflow = FeatureDiscoveryWorkflow(max_iterations=2)

# Use faster model
workflow = FeatureDiscoveryWorkflow(
    llm_provider="anthropic",
    model_name="claude-sonnet-4-5-20250929"  # Faster than Opus
)
```

### Optimize for Quality

```python
# Use more iterations
workflow = FeatureDiscoveryWorkflow(max_iterations=5)

# Increase temperature for creativity
workflow = FeatureDiscoveryWorkflow(temperature=0.9)

# Use more powerful model
workflow = FeatureDiscoveryWorkflow(
    model_name="claude-opus-4-5-20251101"
)
```

### Optimize for Cost

```python
# Use Sonnet instead of Opus
workflow = FeatureDiscoveryWorkflow(
    model_name="claude-sonnet-4-5-20250929"
)

# Limit iterations
workflow = FeatureDiscoveryWorkflow(max_iterations=2)
```

---

## Testing Reference

### Run Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_workflow.py

# With coverage
pytest tests/ --cov=src --cov-report=html

# Verbose output
pytest tests/ -v
```

### Test Structure

```python
import pytest
from src.workflow import FeatureDiscoveryWorkflow

def test_workflow_initialization():
    workflow = FeatureDiscoveryWorkflow()
    assert workflow is not None

@pytest.mark.integration
def test_full_workflow():
    # Integration tests with real API calls
    pass
```

---

## Version History

**v0.1.0** (Current)
- Initial release
- Feature Generator Agent
- Evaluator Agent
- LangGraph workflow orchestration
- CLI interface
- Complete documentation

**Future Versions**
- v0.2.0: Feature Store integration (Phase 2)
- v0.3.0: Ranking Agent (Phase 3)
- v0.4.0: Aggregated features (Phase 4)
