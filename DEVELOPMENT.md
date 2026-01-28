# Development Guide

## System Design Decisions

### Why LangGraph?

**LangGraph** was chosen over other frameworks (CrewAI, AutoGen, raw LangChain) because:

1. **State Management**: Built-in state handling perfect for iterative workflows
2. **Cyclic Graphs**: Natural support for generator-evaluator loops
3. **Flexibility**: Easy to add new nodes (future: search, ranking agents)
4. **Debugging**: Clear state transitions for troubleshooting
5. **Production Ready**: Battle-tested by LangChain team

### Evaluator-Optimizer Pattern

We use an **Evaluator-Optimizer** pattern rather than fully autonomous agents:

**Why this pattern?**
- **Quality over speed**: Iterative refinement produces better features
- **Explainability**: Rubrics provide transparent evaluation criteria
- **Cost effective**: Bounded iterations prevent runaway costs
- **Deterministic**: More predictable than free-form agent collaboration

**Pattern Flow:**
```
Input → Generate → Evaluate → Decision
            ↑          ↓
            └─ Feedback loop ─┘
```

### Agent Design

**Feature Generator Agent:**
- **Role**: Creative generator
- **Prompt Engineering**: Structured output with JSON schema
- **Feedback Integration**: Takes evaluator recommendations
- **Output**: Pydantic-validated features

**Evaluator Agent:**
- **Role**: Critical evaluator + coach
- **Two-stage process**: 
  1. Create problem-specific rubric
  2. Score features + provide feedback
- **Decision logic**: Determines if iteration helps

## Code Architecture

### State Management

```python
WorkflowState = {
    "business_problem": str,          # Input
    "current_features": List[Feature], # Generated features
    "rubric": EvaluationRubric,       # Evaluation criteria
    "evaluations": List[FeatureScore], # Scores
    "iteration": int,                  # Current iteration
    "converged": bool,                 # Stop signal
    "improvement_recommendations": str # Feedback for next round
}
```

State flows through nodes and is updated incrementally.

### Prompt Engineering

**Key principles:**
1. **Structured outputs**: Use JSON schemas for reliability
2. **Domain context**: Banking-specific knowledge
3. **Few-shot examples**: (Can be added in prompts)
4. **Clear constraints**: Explicit rules (8-15 features, raw concepts only)

**Generator Prompt Structure:**
```
System: [Role + task + output schema]
Human: [Business problem + feedback]
```

**Evaluator Prompt Structure:**
```
System: [Role + evaluation criteria]
Human: [Problem + rubric + features + iteration context]
```

### Pydantic Models

**Why Pydantic?**
- Type safety and validation
- Easy serialization/deserialization
- Self-documenting code
- Integration with LangChain parsers

**Model hierarchy:**
```
Feature ─┐
         ├─→ GeneratorOutput
         │
RubricCriterion ─┐
                 ├─→ EvaluationRubric ─┐
                 │                      ├─→ FeatureEvaluation
FeatureScore ────┘                      │
                                        ↓
                                  WorkflowState
```

## Extending the System

### Adding Phase 2: Feature Store Search

**1. Create Vector Search Module:**

```python
# src/search/vector_search.py
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings

class FeatureStoreSearch:
    def __init__(self, qdrant_url, collection_name):
        self.client = QdrantClient(url=qdrant_url)
        self.embeddings = OpenAIEmbeddings()
        self.collection = collection_name
    
    def search(self, features, top_k=10):
        # Embed features
        # Query Qdrant
        # Return matches
        pass
```

**2. Add Search Node to Graph:**

```python
workflow.add_node("search", self._search_node)
workflow.add_edge("finalize", "search")
```

**3. Update State Schema:**

```python
class WorkflowState(BaseModel):
    # ... existing fields ...
    search_results: Optional[List[FeatureMatch]] = None
```

### Adding Phase 3: Ranking Agent

**1. Create Ranking Agent:**

```python
# src/agents/ranker.py
class RankingAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def rank(self, search_results, rubric, context):
        # Re-rank based on:
        # - Semantic similarity
        # - Rubric criteria
        # - Business context
        # - Metadata (freshness, usage)
        pass
```

**2. Add to Workflow:**

```python
workflow.add_node("rank", self._rank_node)
workflow.add_edge("search", "rank")
workflow.add_edge("rank", END)
```

### Adding Aggregated Features

**1. Extend DataType Enum:**

```python
class DataType(str, Enum):
    STRUCTURED = "structured"
    TIME_SERIES = "time_series"
    UNSTRUCTURED = "unstructured"
    EXTERNAL = "external"
    AGGREGATED = "aggregated"  # NEW
```

**2. Update Feature Schema:**

```python
class Feature(BaseModel):
    # ... existing fields ...
    aggregation_type: Optional[str] = None  # sum, avg, count, etc.
    window: Optional[str] = None  # 7d, 30d, 90d
    base_feature: Optional[str] = None  # reference to raw feature
```

**3. Update Generator Prompt:**

Add instructions for generating aggregated features with proper syntax.

## Best Practices

### Prompt Engineering

**DO:**
- Be specific about output format
- Provide clear constraints (e.g., "8-15 features")
- Use banking domain language
- Include rationale fields for explainability

**DON'T:**
- Leave output format ambiguous
- Request too many features at once
- Use generic prompts
- Skip error handling

### Error Handling

**Levels:**
1. **Input validation**: Pydantic models catch schema errors
2. **LLM parsing**: JsonOutputParser handles malformed JSON
3. **Retry logic**: (Can be added) Retry failed API calls
4. **Graceful degradation**: Return partial results if needed

**Example:**

```python
try:
    result = chain.invoke(inputs)
except OutputParserException:
    # Retry with more explicit instructions
    pass
except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise
```

### Testing Strategy

**Unit Tests:**
- Test individual agents with mocked LLMs
- Test state transitions
- Test decision logic

**Integration Tests:**
- Test full workflow with real API calls (mark as slow)
- Test convergence behavior
- Test different business problems

**Example:**

```python
@pytest.mark.slow
@pytest.mark.integration
def test_full_workflow_real_api():
    workflow = FeatureDiscoveryWorkflow()
    result = workflow.run("Predict customer churn")
    assert len(result['features']) >= 8
    assert result['rubric'] is not None
```

### Performance Optimization

**Current bottlenecks:**
1. Sequential LLM calls (generator → evaluator)
2. Multiple iterations

**Optimization strategies:**
1. **Parallel evaluation**: Score features in parallel
2. **Caching**: Cache rubrics for similar problems
3. **Early stopping**: Stop if score improvement < threshold
4. **Batch processing**: Process multiple problems concurrently

**Example:**

```python
# Parallel feature scoring
from concurrent.futures import ThreadPoolExecutor

def score_features_parallel(self, features, rubric):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(self.score_feature, f, rubric) 
            for f in features
        ]
        return [f.result() for f in futures]
```

## Debugging

### Enable Verbose Mode

```python
result = workflow.run(problem, verbose=True)
```

### Inspect State

```python
# Add breakpoint in workflow
def _evaluate_node(self, state):
    print(f"State at iteration {state['iteration']}:")
    print(f"Features: {len(state['current_features'])}")
    import pdb; pdb.set_trace()  # Debug here
    return self.evaluator(state)
```

### LangSmith Tracing

```python
# Enable LangSmith for detailed tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
```

### Common Issues

**Issue: Features not improving across iterations**
- Check evaluator feedback quality
- Increase temperature for more creativity
- Review rubric criteria relevance

**Issue: JSON parsing errors**
- Validate LLM output format
- Add explicit schema examples in prompts
- Check for truncated responses

**Issue: High API costs**
- Reduce max_iterations
- Use smaller models for generation
- Implement caching

## Monitoring & Logging

### Add Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureGeneratorAgent:
    def generate(self, problem, feedback):
        logger.info(f"Generating features for: {problem[:50]}...")
        result = self.chain.invoke(inputs)
        logger.info(f"Generated {len(result.features)} features")
        return result
```

### Track Metrics

```python
class WorkflowMetrics:
    def __init__(self):
        self.iterations = []
        self.scores = []
        self.api_calls = 0
    
    def log_iteration(self, iteration, scores):
        self.iterations.append(iteration)
        self.scores.append(scores)
    
    def summary(self):
        return {
            "total_iterations": len(self.iterations),
            "avg_score_improvement": ...,
            "api_calls": self.api_calls
        }
```

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Keep functions focused (< 50 lines)

### Pull Request Process

1. Create feature branch: `git checkout -b feature/my-feature`
2. Write tests for new code
3. Update documentation
4. Run tests: `pytest tests/`
5. Submit PR with clear description

### Commit Messages

```
feat: Add ranking agent for Phase 3
fix: Handle empty feature lists
docs: Update README with new examples
test: Add integration tests for workflow
```

## Resources

**LangGraph:**
- Docs: https://langchain-ai.github.io/langgraph/
- Examples: https://github.com/langchain-ai/langgraph/tree/main/examples

**LangChain:**
- Docs: https://python.langchain.com/docs/
- Prompt Engineering: https://python.langchain.com/docs/modules/model_io/prompts/

**Qdrant:**
- Docs: https://qdrant.tech/documentation/
- Python Client: https://github.com/qdrant/qdrant-client

## Future Roadmap

**Short-term (1-2 months):**
- [ ] Qdrant Feature Store integration
- [ ] Ranking agent implementation
- [ ] Performance benchmarking
- [ ] Production deployment guide

**Medium-term (3-6 months):**
- [ ] Aggregated feature support
- [ ] Custom taxonomy templates
- [ ] Feature lineage tracking
- [ ] A/B testing framework

**Long-term (6+ months):**
- [ ] Multi-model ensemble (combine GPT + Claude)
- [ ] Automated feature engineering
- [ ] Real-time feature monitoring
- [ ] Feature impact analysis
