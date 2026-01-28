"""
Pydantic models for the Feature Discovery System.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DataType(str, Enum):
    """Types of data features can be derived from."""
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    TIME_SERIES = "time_series"
    EXTERNAL = "external"


class Feature(BaseModel):
    """Represents a candidate feature."""
    name: str = Field(description="Name of the feature")
    description: str = Field(description="Detailed description of what the feature represents")
    data_type: DataType = Field(description="Type of data this feature comes from")
    taxonomy: str = Field(description="Category/taxonomy classification")
    rationale: str = Field(description="Why this feature is relevant to the business problem")


class RubricCriterion(BaseModel):
    """A single criterion in the evaluation rubric."""
    name: str = Field(description="Name of the criterion")
    description: str = Field(description="What this criterion evaluates")
    weight: float = Field(description="Weight/importance of this criterion (0-1)", ge=0, le=1)


class EvaluationRubric(BaseModel):
    """Rubric for evaluating features."""
    criteria: List[RubricCriterion] = Field(description="List of evaluation criteria")
    rationale: str = Field(description="Why this rubric is appropriate for the problem")


class FeatureScore(BaseModel):
    """Score for a single feature."""
    feature_name: str
    criterion_scores: Dict[str, float] = Field(
        description="Scores for each rubric criterion (0-10)"
    )
    overall_score: float = Field(description="Weighted overall score")
    feedback: str = Field(description="Qualitative feedback for improvement")


class FeatureEvaluation(BaseModel):
    """Complete evaluation of all features."""
    rubric: EvaluationRubric
    feature_scores: List[FeatureScore]
    improvement_suggested: bool = Field(
        description="Whether another iteration would improve results"
    )
    improvement_recommendations: Optional[str] = Field(
        default=None,
        description="Specific recommendations for the next iteration"
    )


class GeneratorOutput(BaseModel):
    """Output from the Feature Generator Agent."""
    features: List[Feature]
    taxonomy_explanation: str = Field(
        description="Explanation of the taxonomy structure used"
    )


class WorkflowState(BaseModel):
    """State that flows through the LangGraph workflow."""
    business_problem: str
    current_features: Optional[List[Feature]] = None
    rubric: Optional[EvaluationRubric] = None
    evaluations: Optional[List[FeatureScore]] = None
    iteration: int = 0
    max_iterations: int = 3
    converged: bool = False
    final_output: Optional[Dict] = None
    
    class Config:
        arbitrary_types_allowed = True
