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
    """Represents a candidate feature (full schema)."""
    name: str = Field(description="Name of the feature")
    description: str = Field(description="Detailed description of what the feature represents")
    data_type: DataType = Field(description="Type of data this feature comes from")
    taxonomy: str = Field(description="Category/taxonomy classification")
    rationale: str = Field(description="Why this feature is relevant to the business problem")


class CompactFeature(BaseModel):
    """Represents a candidate feature (compact schema for faster generation)."""
    name: str = Field(description="Feature name")
    desc: str = Field(description="Brief description (max 15 words)")
    type: str = Field(description="Data type: structured, time_series, unstructured, or external")

    def to_feature(self) -> "Feature":
        """Convert compact feature to full Feature."""
        type_map = {
            "structured": DataType.STRUCTURED,
            "time_series": DataType.TIME_SERIES,
            "unstructured": DataType.UNSTRUCTURED,
            "external": DataType.EXTERNAL,
        }
        return Feature(
            name=self.name,
            description=self.desc,
            data_type=type_map.get(self.type, DataType.STRUCTURED),
            taxonomy="",
            rationale=""
        )


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
    overall_score: float = Field(description="Weighted overall score (0-10)", ge=0, le=10)
    feedback: str = Field(description="Qualitative feedback for improvement")


class RawScoreData(BaseModel):
    """
    Normalized intermediate representation of score data from LLM responses.

    This provides a deterministic type for score parsing, handling the various
    formats that different evaluator modes might return.
    """
    name: str = Field(description="Feature name")
    score: Optional[float] = Field(default=None, description="Score value (compact mode)")
    overall_score: Optional[float] = Field(default=None, description="Overall score (standard mode)")
    criterion_scores: Optional[Dict[str, float]] = Field(default=None, description="Per-criterion scores")
    feedback: Optional[str] = Field(default=None, description="Feedback text")


class EvaluationTracker(BaseModel):
    """
    Tracks all feature categories during evaluation for consistent logging.

    This provides a single source of truth for feature counts across
    the evaluation pipeline, ensuring logging consistency.
    """
    # Input tracking
    total_input_features: int = Field(description="Total features received for evaluation")
    carried_features: List[str] = Field(default_factory=list, description="Feature names carried from previous iteration")
    features_to_evaluate: List[str] = Field(default_factory=list, description="Feature names sent to LLM for evaluation")

    # Output tracking
    newly_scored: List[str] = Field(default_factory=list, description="Feature names successfully scored by LLM")
    dropped_no_score: List[str] = Field(default_factory=list, description="Feature names dropped due to missing/invalid LLM score")

    # Final categorization
    kept_above_threshold: List[str] = Field(default_factory=list, description="Feature names above score threshold")
    below_threshold: List[str] = Field(default_factory=list, description="Feature names below score threshold")

    @property
    def total_with_valid_scores(self) -> int:
        """Total features with valid scores (carried + newly scored)."""
        return len(self.carried_features) + len(self.newly_scored)

    @property
    def total_dropped(self) -> int:
        """Total features dropped due to LLM issues."""
        return len(self.dropped_no_score)

    def log_summary(self, iteration: int, max_iterations: int, avg_score: float, threshold: float) -> None:
        """Print a comprehensive evaluation summary."""
        print(f"\n{'='*70}")
        print(f"EVALUATION TRACKING - Iteration {iteration}/{max_iterations}")
        print(f"{'='*70}")

        print(f"\n[INPUT]")
        print(f"  Total features received:     {self.total_input_features}")
        print(f"  Carried from previous iter:  {len(self.carried_features)}")
        print(f"  Sent to LLM for evaluation:  {len(self.features_to_evaluate)}")

        print(f"\n[LLM EVALUATION]")
        print(f"  Successfully scored:         {len(self.newly_scored)}")
        if self.dropped_no_score:
            print(f"  DROPPED (no/invalid score):  {len(self.dropped_no_score)}")
            print(f"    Names: {self.dropped_no_score}")

        print(f"\n[FINAL SCORES]")
        print(f"  Total with valid scores:     {self.total_with_valid_scores}")
        print(f"  Average score:               {avg_score:.2f}")
        print(f"  Above threshold ({threshold}):      {len(self.kept_above_threshold)}")
        print(f"  Below threshold:             {len(self.below_threshold)}")

        if self.kept_above_threshold:
            print(f"\n[KEPT FEATURES]")
            for name in self.kept_above_threshold:
                print(f"    + {name}")

        if self.below_threshold:
            print(f"\n[TO REGENERATE]")
            for name in self.below_threshold:
                print(f"    - {name}")

        print(f"{'='*70}\n")


class CompactFeatureScore(BaseModel):
    """Score for a single feature (compact mode)."""
    name: str
    score: float = Field(description="Overall score 0-10", ge=0, le=10)

    def to_feature_score(self) -> "FeatureScore":
        """Convert to full FeatureScore."""
        return FeatureScore(
            feature_name=self.name,
            criterion_scores={},
            overall_score=self.score,
            feedback=""
        )


class FeatureEvaluation(BaseModel):
    """Complete evaluation of all features."""
    feature_scores: List[FeatureScore]
    improvement_suggested: bool = Field(
        description="Whether another iteration would improve results"
    )
    improvement_recommendations: Optional[str] = Field(
        default=None,
        description="Specific recommendations for the next iteration"
    )


class CompactFeatureEvaluation(BaseModel):
    """Complete evaluation of all features (compact mode)."""
    scores: List[CompactFeatureScore]
    continue_: bool = Field(alias="continue", description="Whether to continue iterating")
    feedback: Optional[str] = Field(default=None, description="Brief feedback for next iteration")

    model_config = {"populate_by_name": True}


class GeneratorOutput(BaseModel):
    """Output from the Feature Generator Agent."""
    features: List[Feature]
    taxonomy_explanation: str = Field(
        description="Explanation of the taxonomy structure used"
    )


class CompactGeneratorOutput(BaseModel):
    """Output from the Feature Generator Agent (compact mode)."""
    features: List[CompactFeature]


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
