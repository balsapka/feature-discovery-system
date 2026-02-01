"""
Evaluator Agent - Creates rubrics and evaluates generated features.
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain_core.exceptions import OutputParserException
from ..models.schemas import (
    Feature,
    EvaluationRubric,
    RubricCriterion,
    FeatureEvaluation,
    FeatureScore,
    RawScoreData,
    EvaluationTracker,
)

# Configure module logger
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation fails in a way that cannot be recovered."""
    pass


@dataclass
class BatchResult:
    """Result from evaluating a batch of features."""
    scores: List[FeatureScore] = field(default_factory=list)
    scored_names: List[str] = field(default_factory=list)
    dropped_names: List[str] = field(default_factory=list)


class RobustJsonOutputParser(BaseOutputParser[dict]):
    """
    A more robust JSON parser that can handle malformed JSON,
    especially truncated arrays at the end.
    """

    def parse(self, text: str) -> dict:
        """Parse LLM output, attempting to fix common JSON issues."""
        # First try standard parsing
        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                text = json_match.group(1)

            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to fix common issues
        fixed_text = self._fix_json(text)
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError as e:
            raise OutputParserException(f"Failed to parse JSON: {e}\nText: {text[:500]}")

    def _fix_json(self, text: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        text = text.strip()

        # Extract JSON object if wrapped in other text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)

        # Fix trailing commas in arrays (common LLM issue)
        text = re.sub(r',\s*\]', ']', text)
        text = re.sub(r',\s*\}', '}', text)

        # Try to fix truncated array elements
        text = re.sub(r',\s*\{[^}]*$', '', text)

        # Ensure proper closing brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')

        text += '}' * max(0, open_braces)
        text += ']' * max(0, open_brackets)

        return text

    @property
    def _type(self) -> str:
        return "robust_json"


class BaseEvaluator(ABC):
    """
    Abstract base class for feature evaluators.

    Provides common functionality for evaluating features against rubrics,
    including batching, parallel execution, and score parsing.
    """

    def __init__(
        self,
        llm,
        parallel: bool = False,
        score_threshold: float = 7.0,
        batch_size: int = 10
    ):
        """
        Initialize the base evaluator.

        Args:
            llm: Language model instance (OpenAI or Anthropic)
            parallel: If True, evaluate batches in parallel for faster execution
            score_threshold: Minimum average score to stop iterating (default 7.0)
            batch_size: Number of features per evaluation batch (default 10)
        """
        self.llm = llm
        self.parallel = parallel
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self._setup_prompts_and_parsers()

    @abstractmethod
    def _setup_prompts_and_parsers(self) -> None:
        """Set up evaluation prompts and output parsers."""
        pass

    @abstractmethod
    def create_rubric(self, business_problem: str) -> EvaluationRubric:
        """Create an evaluation rubric for the business problem."""
        pass

    @abstractmethod
    def _format_features_for_prompt(self, features: List[Feature], rubric: Optional[EvaluationRubric]) -> Dict[str, str]:
        """Format features and rubric for the evaluation prompt."""
        pass

    @abstractmethod
    def _extract_raw_scores(self, result: Dict[str, Any]) -> List[RawScoreData]:
        """Extract raw score data from LLM response."""
        pass

    def _parse_score(self, score_data: RawScoreData, feature_name: str) -> Optional[FeatureScore]:
        """
        Parse a RawScoreData into a FeatureScore.

        Returns None if the score cannot be validated.
        """
        try:
            score_val = score_data.score if score_data.score is not None else score_data.overall_score
            if score_val is None:
                logger.debug(f"No score field for '{feature_name}', skipping")
                return None

            score_float = float(score_val)

            # Validate range (0-10 as defined in prompts)
            if score_float < 0 or score_float > 10:
                logger.debug(f"Score {score_float} out of range for '{feature_name}', skipping")
                return None

            return FeatureScore(
                feature_name=feature_name,
                criterion_scores=score_data.criterion_scores or {},
                overall_score=score_float,
                feedback=score_data.feedback or ""
            )
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse score for '{feature_name}': {e}")
            return None

    def _evaluate_batch(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: Optional[EvaluationRubric],
        iteration: int,
        max_iterations: int
    ) -> BatchResult:
        """
        Evaluate a batch of features.

        Returns:
            BatchResult with scores and tracking information
        """
        result = BatchResult()

        prompt_data = self._format_features_for_prompt(features, rubric)

        chain = self.evaluation_prompt | self.llm | self.evaluation_parser
        llm_result = chain.invoke({
            "business_problem": business_problem,
            **prompt_data,
            "iteration": iteration + 1,
            "max_iterations": max_iterations
        })

        raw_scores = self._extract_raw_scores(llm_result)

        # Build a map of scores by name for reliable matching
        score_by_name = {score.name: score for score in raw_scores if score.name}

        # Parse scores using name-based matching
        for feature in features:
            score_data = score_by_name.get(feature.name)
            if score_data:
                score = self._parse_score(score_data, feature.name)
                if score:
                    result.scores.append(score)
                    result.scored_names.append(feature.name)
                else:
                    result.dropped_names.append(feature.name)
                    logger.debug(f"Could not parse score for '{feature.name}'")
            else:
                result.dropped_names.append(feature.name)
                logger.warning(f"No score returned for feature '{feature.name}'")

        return result

    def evaluate_features(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: EvaluationRubric,
        iteration: int,
        max_iterations: int,
        previous_scores: Optional[Dict[str, FeatureScore]] = None
    ) -> Tuple[FeatureEvaluation, EvaluationTracker]:
        """
        Evaluate features against the rubric.

        Returns:
            Tuple of (FeatureEvaluation, EvaluationTracker)
        """
        if not business_problem or not business_problem.strip():
            raise EvaluationError("Business problem cannot be empty")
        if not features:
            raise EvaluationError("Features list cannot be empty")

        # Initialize tracker
        tracker = EvaluationTracker(total_input_features=len(features))

        # Separate features needing evaluation vs carried forward
        features_to_evaluate = []
        carried_scores = []

        if previous_scores:
            for feature in features:
                if feature.name in previous_scores:
                    carried_scores.append(previous_scores[feature.name])
                    tracker.carried_features.append(feature.name)
                else:
                    features_to_evaluate.append(feature)
                    tracker.features_to_evaluate.append(feature.name)
        else:
            features_to_evaluate = features
            tracker.features_to_evaluate = [f.name for f in features]

        # Evaluate new features
        all_scores = []
        if features_to_evaluate:
            batch_result = self._evaluate_with_batching(
                business_problem, features_to_evaluate, rubric, iteration, max_iterations
            )
            all_scores = batch_result.scores
            tracker.newly_scored = batch_result.scored_names
            tracker.dropped_no_score = batch_result.dropped_names

        # Combine with carried scores
        combined_scores = carried_scores + all_scores

        if not combined_scores:
            raise EvaluationError("No valid scores could be parsed from evaluation")

        # Calculate decision
        avg_score = sum(fs.overall_score for fs in combined_scores) / len(combined_scores)
        should_continue = avg_score < self.score_threshold and iteration < max_iterations - 1

        return FeatureEvaluation(
            feature_scores=combined_scores,
            improvement_suggested=should_continue,
            improvement_recommendations="Continue improving low-scoring features" if should_continue else None
        ), tracker

    def _evaluate_with_batching(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: Optional[EvaluationRubric],
        iteration: int,
        max_iterations: int
    ) -> BatchResult:
        """Handle batching logic for feature evaluation."""
        if len(features) > self.batch_size:
            batches = [
                features[i:i + self.batch_size]
                for i in range(0, len(features), self.batch_size)
            ]
            logger.info(f"Evaluating {len(features)} features in {len(batches)} batches of up to {self.batch_size}")

            if self.parallel and len(batches) > 1:
                return self._evaluate_batches_parallel(
                    business_problem, batches, rubric, iteration, max_iterations
                )
            else:
                return self._evaluate_batches_sequential(
                    business_problem, batches, rubric, iteration, max_iterations
                )
        else:
            return self._evaluate_batch(
                business_problem, features, rubric, iteration, max_iterations
            )

    def _evaluate_batches_parallel(
        self,
        business_problem: str,
        batches: List[List[Feature]],
        rubric: Optional[EvaluationRubric],
        iteration: int,
        max_iterations: int
    ) -> BatchResult:
        """Evaluate batches in parallel."""
        logger.info(f"Using parallel evaluation with {len(batches)} workers")
        combined = BatchResult()
        failed_batches = []

        with ThreadPoolExecutor(max_workers=len(batches)) as executor:
            futures = {
                executor.submit(
                    self._evaluate_batch,
                    business_problem, batch, rubric, iteration, max_iterations
                ): (i, batch) for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx, batch = futures[future]
                try:
                    result = future.result()
                    combined.scores.extend(result.scores)
                    combined.scored_names.extend(result.scored_names)
                    combined.dropped_names.extend(result.dropped_names)
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} failed in parallel mode: {e}, will retry sequentially")
                    failed_batches.append(batch)

        # Retry failed batches sequentially
        if failed_batches:
            logger.info(f"Retrying {len(failed_batches)} failed batches sequentially")
            for batch in failed_batches:
                try:
                    result = self._evaluate_batch(
                        business_problem, batch, rubric, iteration, max_iterations
                    )
                    combined.scores.extend(result.scores)
                    combined.scored_names.extend(result.scored_names)
                    combined.dropped_names.extend(result.dropped_names)
                except Exception as e:
                    logger.error(f"Batch retry also failed: {e}")
                    # All features in this batch are dropped
                    combined.dropped_names.extend([f.name for f in batch])

        return combined

    def _evaluate_batches_sequential(
        self,
        business_problem: str,
        batches: List[List[Feature]],
        rubric: Optional[EvaluationRubric],
        iteration: int,
        max_iterations: int
    ) -> BatchResult:
        """Evaluate batches sequentially."""
        combined = BatchResult()

        for batch in batches:
            result = self._evaluate_batch(
                business_problem, batch, rubric, iteration, max_iterations
            )
            combined.scores.extend(result.scores)
            combined.scored_names.extend(result.scored_names)
            combined.dropped_names.extend(result.dropped_names)

        return combined

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node function."""
        business_problem = state["business_problem"]
        features = state["current_features"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]

        if not features:
            raise EvaluationError("No features to evaluate in state")

        # Create rubric on first iteration
        rubric = state.get("rubric") or self.create_rubric(business_problem)

        # Build previous scores for token optimization
        previous_scores = self._build_previous_scores(state)

        # Evaluate
        evaluation, tracker = self.evaluate_features(
            business_problem, features, rubric, iteration, max_iterations,
            previous_scores=previous_scores
        )

        # Get previously rejected names to accumulate across iterations
        previous_rejected_names = state.get("rejected_feature_names", set())

        return self._build_output_state(
            features, evaluation, tracker, rubric, iteration, max_iterations,
            previous_rejected_names=previous_rejected_names
        )

    def _build_previous_scores(self, state: Dict[str, Any]) -> Optional[Dict[str, FeatureScore]]:
        """Build previous scores map from state."""
        kept_feature_names = state.get("kept_feature_names")
        if kept_feature_names is None:
            kept_features = state.get("kept_features", [])
            if kept_features:
                kept_feature_names = {f.name for f in kept_features}

        if kept_feature_names and state.get("evaluations"):
            previous_scores = {}
            for score in state["evaluations"]:
                if score.feature_name in kept_feature_names:
                    previous_scores[score.feature_name] = score
            logger.debug(f"Built previous_scores with {len(previous_scores)} entries from {len(kept_feature_names)} kept names")
            return previous_scores
        return None

    def _build_output_state(
        self,
        features: List[Feature],
        evaluation: FeatureEvaluation,
        tracker: EvaluationTracker,
        rubric: EvaluationRubric,
        iteration: int,
        max_iterations: int,
        previous_rejected_names: set = None
    ) -> Dict[str, Any]:
        """Build the output state dictionary."""
        # Build a map of feature name -> feature for lookup
        feature_map = {f.name: f for f in features}

        # Categorize features based on scores
        kept_features = []
        low_scoring_features = []

        for score in evaluation.feature_scores:
            feature = feature_map.get(score.feature_name)
            if not feature:
                # This shouldn't happen, but handle gracefully
                logger.warning(f"Score for unknown feature: {score.feature_name}")
                continue

            if score.overall_score >= self.score_threshold:
                kept_features.append(feature)
                tracker.kept_above_threshold.append(feature.name)
            else:
                low_scoring_features.append(feature)
                tracker.below_threshold.append(feature.name)

        low_scoring_count = len(low_scoring_features)

        # Calculate average score for logging
        avg_score = sum(s.overall_score for s in evaluation.feature_scores) / len(evaluation.feature_scores)

        # Log comprehensive summary
        tracker.log_summary(iteration + 1, max_iterations, avg_score, self.score_threshold)

        # Continue if there are low-scoring features and not at max iterations
        should_continue = low_scoring_count > 0 and iteration < max_iterations - 1

        # Build recommendations
        low_scoring_type_counts = {}
        if should_continue and low_scoring_features:
            for f in low_scoring_features:
                dtype = str(f.data_type.value) if hasattr(f.data_type, 'value') else str(f.data_type)
                low_scoring_type_counts[dtype] = low_scoring_type_counts.get(dtype, 0) + 1

            type_info = ", ".join([f"{count} {dtype}" for dtype, count in low_scoring_type_counts.items()])
            recommendations = f"Replace {low_scoring_count} low-scoring features ({type_info}). "
            recommendations += "Maintain data type variety. "
            if evaluation.improvement_recommendations:
                recommendations += evaluation.improvement_recommendations
        else:
            recommendations = evaluation.improvement_recommendations

        # Track rejected feature names to prevent regenerating them
        rejected_names = {f.name for f in low_scoring_features}
        # Also include any previously rejected names
        all_rejected_names = rejected_names | (previous_rejected_names or set())

        return {
            "rubric": rubric,
            "evaluations": evaluation.feature_scores,
            "improvement_recommendations": recommendations,
            "kept_features": kept_features,
            "low_scoring_count": low_scoring_count,
            "low_scoring_type_counts": low_scoring_type_counts or None,
            "rejected_feature_names": all_rejected_names,
            "converged": not should_continue,
            "iteration": iteration + 1
        }


class StandardEvaluator(BaseEvaluator):
    """
    Standard evaluator with full rubric generation and detailed scoring.

    Uses complete feature schemas and generates problem-specific rubrics.
    """

    def _setup_prompts_and_parsers(self) -> None:
        """Set up evaluation prompts and output parsers."""
        self.rubric_parser = JsonOutputParser(pydantic_object=EvaluationRubric)
        self.evaluation_parser = JsonOutputParser(pydantic_object=FeatureEvaluation)
        self.rubric_prompt = self._create_rubric_prompt()
        self.evaluation_prompt = self._create_evaluation_prompt()

    def _create_rubric_prompt(self) -> ChatPromptTemplate:
        """Create prompt for rubric generation."""
        system_message = """You are an expert evaluator for data science features in banking.

Your task is to create a problem-specific evaluation rubric that will be used to score candidate features.

The rubric should include 4-6 criteria relevant to the business problem. Common criteria include:
- Relevance: How directly the feature relates to the problem
- Data Availability: Likelihood the data exists in a banking system
- Predictive Power: Expected impact on model performance
- Regulatory Compliance: Alignment with banking regulations
- Computational Feasibility: Ease of calculation/storage
- Business Interpretability: How easily stakeholders can understand it

Each criterion should have:
- A clear name
- A description of what it evaluates
- A weight (0-1) indicating its importance (weights should sum to 1.0)

IMPORTANT: Return valid JSON matching this schema:
{{
    "criteria": [
        {{
            "name": "criterion_name",
            "description": "what this evaluates",
            "weight": 0.25
        }}
    ],
    "rationale": "why this rubric is appropriate for this problem"
}}"""

        human_message = """Business Problem:
{business_problem}

Create an evaluation rubric for this problem as JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _create_evaluation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for feature evaluation."""
        system_message = """You are an expert feature evaluator for banking data science projects.

Your task is to:
1. Score each feature against the rubric criteria (scale 0-10)
2. Calculate weighted overall scores
3. Provide constructive feedback for improvement
4. Determine if another iteration would meaningfully improve the features

Be rigorous but fair. Look for:
- Missing important feature categories
- Poorly defined or vague features
- Features that don't align well with the business problem
- Opportunities to add diversity in data types
- Regulatory or compliance concerns

IMPORTANT: Return valid JSON matching this schema:
{{
    "feature_scores": [
        {{
            "feature_name": "name",
            "criterion_scores": {{"criterion1": 8.0, "criterion2": 7.5}},
            "overall_score": 7.8,
            "feedback": "specific feedback for this feature"
        }}
    ],
    "improvement_suggested": true or false,
    "improvement_recommendations": "overall recommendations for next iteration (if improvement_suggested is true)"
}}"""

        human_message = """Business Problem:
{business_problem}

Evaluation Rubric:
{rubric}

Features to Evaluate:
{features}

Current Iteration: {iteration} of {max_iterations}

Evaluate these features and determine if another iteration would help."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def create_rubric(self, business_problem: str) -> EvaluationRubric:
        """Create an evaluation rubric for the business problem."""
        chain = self.rubric_prompt | self.llm | self.rubric_parser
        result = chain.invoke({"business_problem": business_problem})
        return EvaluationRubric(**result)

    def _format_features_for_prompt(self, features: List[Feature], rubric: Optional[EvaluationRubric]) -> Dict[str, str]:
        """Format features and rubric for the evaluation prompt."""
        features_text = "\n\n".join([
            f"Feature: {f.name}\n"
            f"Description: {f.description}\n"
            f"Data Type: {f.data_type}\n"
            f"Taxonomy: {f.taxonomy}\n"
            f"Rationale: {f.rationale}"
            for f in features
        ])

        rubric_text = f"Criteria:\n" + "\n".join([
            f"- {c.name} (weight: {c.weight}): {c.description}"
            for c in rubric.criteria
        ])

        return {"features": features_text, "rubric": rubric_text}

    def _extract_raw_scores(self, result: Dict[str, Any]) -> List[RawScoreData]:
        """Extract raw score data from LLM response."""
        raw_scores = result.get("feature_scores", [])
        return [
            RawScoreData(
                name=s.get("feature_name", ""),
                score=s.get("overall_score"),
                overall_score=s.get("overall_score"),
                criterion_scores=s.get("criterion_scores", {}),
                feedback=s.get("feedback", "")
            )
            for s in raw_scores
            if isinstance(s, dict)
        ]


class CompactEvaluator(BaseEvaluator):
    """
    Compact evaluator for faster evaluation with minimal schema.

    Uses simplified scoring without detailed rubrics for faster iteration.
    """

    def _setup_prompts_and_parsers(self) -> None:
        """Set up evaluation prompts and output parsers."""
        self.evaluation_parser = RobustJsonOutputParser()
        self.evaluation_prompt = self._create_evaluation_prompt()
        self.default_rubric = EvaluationRubric(
            criteria=[
                RubricCriterion(name="Relevance", description="Feature relevance to problem", weight=0.4),
                RubricCriterion(name="Feasibility", description="Data availability and compute cost", weight=0.3),
                RubricCriterion(name="Predictive", description="Expected predictive power", weight=0.3),
            ],
            rationale="Default compact rubric"
        )

    def _create_evaluation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for feature evaluation (compact mode)."""
        system_message = """Score ALL features 0-10. You MUST return a score for EVERY feature listed.
Output JSON:
{{
    "scores": [{{"name": "feature_name", "score": 7.5}}],
    "continue": true or false,
    "feedback": "brief feedback if continue=true"
}}

CRITICAL: Include EVERY feature name exactly as provided. Do not skip any features.
Set continue=false if scores are good (avg > 7) or max iterations reached."""

        human_message = """Problem: {business_problem}
Features to score: {features}
Iteration: {iteration}/{max_iterations}

Score ALL features as JSON. Return one score object for each feature name listed above."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def create_rubric(self, business_problem: str) -> EvaluationRubric:
        """Return the default compact rubric."""
        return self.default_rubric

    def _format_features_for_prompt(self, features: List[Feature], rubric: Optional[EvaluationRubric]) -> Dict[str, str]:
        """Format features for the compact evaluation prompt."""
        features_text = ", ".join([f.name for f in features])
        return {"features": features_text}

    def _extract_raw_scores(self, result: Dict[str, Any]) -> List[RawScoreData]:
        """Extract raw score data from LLM response."""
        raw_scores = result.get("scores", result.get("feature_scores", []))
        return [
            RawScoreData(
                name=s.get("name", ""),
                score=s.get("score"),
                overall_score=s.get("score"),
                criterion_scores={},
                feedback=""
            )
            for s in raw_scores
            if isinstance(s, dict)
        ]


# Factory function for backwards compatibility
def EvaluatorAgent(
    llm,
    compact_mode: bool = False,
    parallel: bool = False,
    score_threshold: float = 7.0,
    batch_size: int = 10
) -> BaseEvaluator:
    """
    Factory function to create the appropriate evaluator.

    Args:
        llm: Language model instance (OpenAI or Anthropic)
        compact_mode: If True, use CompactEvaluator for faster evaluation
        parallel: If True, evaluate batches in parallel for faster execution
        score_threshold: Minimum average score to stop iterating (default 7.0)
        batch_size: Number of features per evaluation batch (default 10)

    Returns:
        BaseEvaluator: Either StandardEvaluator or CompactEvaluator instance
    """
    if compact_mode:
        return CompactEvaluator(
            llm=llm,
            parallel=parallel,
            score_threshold=score_threshold,
            batch_size=batch_size
        )
    return StandardEvaluator(
        llm=llm,
        parallel=parallel,
        score_threshold=score_threshold,
        batch_size=batch_size
    )
