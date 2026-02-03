# Model Recommendations for Feature Discovery Workflow

Based on benchmark research, here are my rigorous recommendations for each component:

## Summary Table

| Agent/Task | Recommended Model | Alternative | Rationale |
|------------|-------------------|-------------|-----------|
| Summarizer | qwen3-32b-v1:0 | meta.llama3-70b-instruct | Balanced cost/performance, strong instruction following |
| Feature Generator | deepseek.v3-v1:0 | moonshot.kimi-k2-thinking | Best reasoning + creativity at efficient cost |
| Evaluator | qwen-235b-a22b-2507-v1:0 | mistral-large-3-675b-instruct | Highest analytical reasoning, consistent scoring |
| Embedder | Qwen3Embedding8b | BgeLargeEmbeddings | SOTA on MTEB retrieval; BGE as fast fallback |
| Ranking/Top-K Agent | qwen3-32b-v1:0 | apac.amazon.nova-pro-v1:0 | Good reasoning + fast for final selection |

## Detailed Analysis by Task

### 1. Summarizer Agent

**Task:** Condense schema, sample rows, metadata into structured summary

| Model | MMLU | IFEval | Structured Output | Cost | Recommendation |
|-------|------|--------|-------------------|------|----------------|
| qwen3-32b-v1:0 | 68.7 (Pro) | High | Excellent | Low | ✅ Best Pick |
| meta.llama3-70b-instruct | 86.0 | 92.1 | Good | Medium | Good alternative |
| apac.amazon.nova-lite-v1:0 | ~80 | Good | Good | Very Low | Budget option |

**Recommendation: qwen3-32b-v1:0**
- Strong instruction following for structured extraction
- Supports thinking/non-thinking modes (use non-thinking for speed)
- Cost-effective for a preprocessing step
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

### 2. Feature Generator Agent

**Task:** Creative generation of relevant feature engineering ideas with reasoning

| Model | MMLU | GPQA | Math/Reasoning | Creativity | Cost | Recommendation |
|-------|------|------|----------------|------------|------|----------------|
| deepseek.v3-v1:0 | 88.5 | 59.1 | SOTA | High | Low | ✅ Best Pick |
| moonshot.kimi-k2-thinking | ~85 | 85.7 | SOTA | Very High | Medium | Best quality |
| qwen-235b-a22b-2507-v1:0 | 76.6+ | High | 85.1 AIME | High | Medium | Strong alternative |

**Recommendation: deepseek.v3-v1:0**
- Best price-performance ratio for reasoning tasks
- 88.5 MMLU, 75.9 MMLU-Pro - strong domain knowledge
- Excellent at generating novel ideas with solid reasoning
- [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437)

**If quality is paramount:** Use moonshot.kimi-k2-thinking
- 85.7% GPQA Diamond (beats GPT-5's 84.5%)
- Native thinking mode with 200-300 step reasoning chains
- [Kimi K2 GitHub](https://github.com/MoonshotAI/Kimi-K2)

### 3. Evaluator Agent

**Task:** Analytical scoring and critique against rubric criteria

| Model | MMLU-Pro | GPQA | Consistency | Analytical | Cost |
|-------|----------|------|-------------|------------|------|
| qwen-235b-a22b-2507-v1:0 | High | High | Very High | Excellent | Medium |
| mistral-large-3-675b-instruct | 78% | 71.2% | High | Excellent | Medium |
| deepseek.v3-v1:0 | 75.9 | 59.1 | High | Good | Low |

**Recommendation: qwen-235b-a22b-2507-v1:0**
- Thinking mode ensures step-by-step evaluation reasoning
- 85.1 AIME score shows strong analytical capability
- Consistent scoring behavior (critical for evaluator-optimizer loop)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)

**Rationale for flagship model here:** The evaluator is the "gatekeeper" of quality. Using a stronger model here ensures:
- Consistent scoring across iterations
- Detailed, actionable feedback for the generator
- Reliable convergence of the optimization loop

### 4. Embedder (for Feature Matching)

**Task:** Embed feature name+description for semantic similarity search

| Model | MTEB Overall | MTEB Retrieval | Multilingual | Speed |
|-------|--------------|----------------|--------------|-------|
| Qwen3Embedding8b | 70.58 (multilingual) | SOTA | 100+ languages | Medium |
| BgeLargeEmbeddings | Good | 83.1 STS | 100+ languages | Fast |

**Recommendation: Qwen3Embedding8b**
- #1 on MTEB Multilingual leaderboard
- 80.68 on MTEB Code benchmark (excellent for technical feature names)
- Outperforms Gemini-Embedding on retrieval tasks
- [Qwen3 Embedding Paper](https://arxiv.org/abs/2506.05176)

**When to use BgeLargeEmbeddings:**
- If latency is critical (faster inference)
- For simpler feature banks with less technical jargon
- [BGE-M3 Comparison](https://huggingface.co/BAAI/bge-m3)

### 5. Ranking/Top-K Select Agent

**Task:** Final intelligent selection from similarity results

| Model | Reasoning | Speed | Structured Output | Cost |
|-------|-----------|-------|-------------------|------|
| qwen3-32b-v1:0 | Good | Fast | Excellent | Low |
| apac.amazon.nova-pro-v1:0 | 85.9 MMLU | Fast | Good | Low |
| meta.llama3-70b-instruct | 86.0 MMLU | Medium | Good | Medium |

**Recommendation: qwen3-32b-v1:0 (non-thinking mode)**
- Fast inference for final selection step
- Good at comparing/ranking candidates
- Reliable JSON/structured output
- Same model as Summarizer = fewer model loads

## Cost-Optimized Configuration

If minimizing cost while maintaining quality:

```python
MODEL_CONFIG = {
    "summarizer": "apac.amazon.nova-lite-v1:0",      # Cheapest, still capable
    "generator": "deepseek.v3-v1:0",                  # Best value for reasoning
    "evaluator": "qwen3-32b-v1:0",                    # Good enough, much cheaper
    "embedder": "BgeLargeEmbeddings",                 # Faster, still effective
    "ranker": "apac.amazon.nova-micro-v1:0",          # Simple task, use smallest
}
```

## Quality-Optimized Configuration

If maximizing output quality:

```python
MODEL_CONFIG = {
    "summarizer": "qwen3-32b-v1:0",                   # Solid extraction
    "generator": "moonshot.kimi-k2-thinking",         # Best reasoning + creativity
    "evaluator": "qwen-235b-a22b-2507-v1:0",          # Flagship for critical evaluation
    "embedder": "Qwen3Embedding8b",                   # SOTA retrieval
    "ranker": "deepseek.v3-v1:0",                     # Strong comparison ability
}
```

## Sources

- [Qwen3 Technical Report (arXiv)](https://arxiv.org/abs/2505.09388)
- [DeepSeek V3 Technical Report (arXiv)](https://arxiv.org/abs/2412.19437)
- [Kimi K2 GitHub](https://github.com/MoonshotAI/Kimi-K2)
- [Mistral Large 3 (HuggingFace)](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411)
- [Amazon Nova Technical Report](https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html)
- [Qwen3 Embedding Paper (arXiv)](https://arxiv.org/abs/2506.05176)
- [BGE vs Qwen3 Embedding Comparison](https://huggingface.co/BAAI/bge-m3)
- [Llama 3.3 70B (HuggingFace)](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [MTEB Leaderboard Analysis](https://huggingface.co/spaces/mteb/leaderboard)
