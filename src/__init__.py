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

__all__ = [
    'FeatureDiscoveryWorkflow',
    'FeatureGeneratorAgent',
    'EvaluatorAgent',
    'DataType',
    'Feature',
    'RubricCriterion',
    'EvaluationRubric',
    'FeatureScore',
    'FeatureEvaluation',
    'GeneratorOutput',
    'WorkflowState'
]

__version__ = '0.1.0'
