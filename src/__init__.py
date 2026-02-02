"""Feature Discovery System package."""
from .workflow import FeatureDiscoveryWorkflow
from .agents import FeatureGeneratorAgent, EvaluatorAgent
from .models import (
    DataType,
    Feature,
    RubricCriterion,
    EvaluationRubric,
    FeatureScore,
    FeatureEvaluation,
    GeneratorOutput,
    WorkflowState
)
from .feature_matcher import (
    create_matcher,
    BaseFeatureMatcher,
    InMemoryVectorMatcher,
    FuzzyMatcher,
    TFIDFMatcher,
    HybridMatcher,
    QdrantMatcher,
    FeatureMatch,
    FeatureMatchResult,
)

__all__ = [
    # Workflow
    'FeatureDiscoveryWorkflow',
    # Agents
    'FeatureGeneratorAgent',
    'EvaluatorAgent',
    # Models
    'DataType',
    'Feature',
    'RubricCriterion',
    'EvaluationRubric',
    'FeatureScore',
    'FeatureEvaluation',
    'GeneratorOutput',
    'WorkflowState',
    # Feature Matching
    'create_matcher',
    'BaseFeatureMatcher',
    'InMemoryVectorMatcher',
    'FuzzyMatcher',
    'TFIDFMatcher',
    'HybridMatcher',
    'QdrantMatcher',
    'FeatureMatch',
    'FeatureMatchResult',
]

__version__ = '0.1.0'
