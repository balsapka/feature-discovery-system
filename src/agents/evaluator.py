"""
Evaluator Agent - Creates rubrics and evaluates generated features.
"""
import json
import logging
import re
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
)

# Configure module logger
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation fails in a way that cannot be recovered."""
    pass


class ScoreParseError(Exception):
    """Raised when score parsing fails for a specific feature."""
    def __init__(self, feature_name: str, reason: str, raw_data: Any = None):
        self.feature_name = feature_name
        self.reason = reason
        self.raw_data = raw_data
        super().__init__(f"Failed to parse score for '{feature_name}': {reason}")


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
        # Remove any leading/trailing whitespace
        text = text.strip()

        # Extract JSON object if wrapped in other text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)

        # Fix trailing commas in arrays (common LLM issue)
        text = re.sub(r',\s*\]', ']', text)
        text = re.sub(r',\s*\}', '}', text)

        # Try to fix truncated array elements - remove incomplete last element
        # Pattern: looks for incomplete object at end of array
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


class EvaluatorAgent:
    """
    Agent responsible for creating evaluation rubrics and scoring features.
    Uses an evaluator-optimizer pattern to iteratively improve feature quality.
    """

    # Valid score range
    MIN_SCORE = 0.0
    MAX_SCORE = 10.0
    DEFAULT_BATCH_SIZE = 10

    def __init__(
        self,
        llm,
        compact_mode: bool = False,
        score_threshold: float = 7.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        strict_mode: bool = False
    ):
        """
        Initialize the Evaluator Agent.

        Args:
            llm: Language model instance (OpenAI or Anthropic)
            compact_mode: If True, use minimal schema for faster evaluation
            score_threshold: Minimum average score to stop iterating (default 7.0)
            batch_size: Number of features per evaluation batch (default 10)
            strict_mode: If True, raise exceptions on parse errors instead of using fallbacks
        """
        self.llm = llm
        self.compact_mode = compact_mode
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.strict_mode = strict_mode

        if compact_mode:
            # Use robust parser for compact mode to handle malformed JSON
            self.evaluation_parser = RobustJsonOutputParser()
            self.evaluation_prompt = self._create_compact_evaluation_prompt()
            # In compact mode, use a simple default rubric instead of generating one
            self.default_rubric = EvaluationRubric(
                criteria=[
                    RubricCriterion(name="Relevance", description="Feature relevance to problem", weight=0.4),
                    RubricCriterion(name="Feasibility", description="Data availability and compute cost", weight=0.3),
                    RubricCriterion(name="Predictive", description="Expected predictive power", weight=0.3),
                ],
                rationale="Default compact rubric"
            )
        else:
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
        """Create prompt for feature evaluation (full mode)."""

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

    def _create_compact_evaluation_prompt(self) -> ChatPromptTemplate:
        """Create prompt for feature evaluation (compact mode)."""

        system_message = """Score features 0-10. Output JSON:
{{
    "scores": [{{"name": "feature_name", "score": 7.5}}],
    "continue": true or false,
    "feedback": "brief feedback if continue=true"
}}

Set continue=false if scores are good (avg > 7) or max iterations reached."""

        human_message = """Problem: {business_problem}
Features: {features}
Iteration: {iteration}/{max_iterations}

Score as JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _validate_score(self, score: Any, feature_name: str) -> Tuple[float, Optional[str]]:
        """
        Validate and normalize a score value.

        Args:
            score: The raw score value from LLM
            feature_name: Name of the feature (for error messages)

        Returns:
            Tuple of (validated_score, error_message_or_none)
        """
        error_msg = None

        # Try to convert to float
        try:
            score_float = float(score)
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid score type '{type(score).__name__}': {score}"
            logger.warning(f"Score validation failed for '{feature_name}': {error_msg}")
            if self.strict_mode:
                raise ScoreParseError(feature_name, error_msg, score)
            return 5.0, error_msg

        # Check range
        if score_float < self.MIN_SCORE or score_float > self.MAX_SCORE:
            error_msg = f"Score {score_float} out of range [{self.MIN_SCORE}, {self.MAX_SCORE}]"
            logger.warning(f"Score out of range for '{feature_name}': {error_msg}")
            # Clamp to valid range instead of using arbitrary default
            score_float = max(self.MIN_SCORE, min(self.MAX_SCORE, score_float))

        return score_float, error_msg

    def _parse_single_score(
        self,
        score_data: Any,
        feature: Feature,
        index: int
    ) -> FeatureScore:
        """
        Parse a single score from LLM output with proper error tracking.

        This is the single source of truth for score parsing logic.

        Args:
            score_data: Raw score data from LLM
            feature: The feature being scored
            index: Position index (for logging)

        Returns:
            FeatureScore with parse_error set if there were issues
        """
        parse_error = None
        criterion_scores = {}
        overall_score = 5.0  # Will be overwritten if parsing succeeds
        feedback = ""

        if isinstance(score_data, dict):
            # Try different field names for score
            score_val = score_data.get("score") or score_data.get("overall_score")

            if score_val is None:
                parse_error = "No 'score' or 'overall_score' field found"
                logger.warning(f"[{index}] {feature.name}: {parse_error}")
                if self.strict_mode:
                    raise ScoreParseError(feature.name, parse_error, score_data)
            else:
                overall_score, validation_error = self._validate_score(score_val, feature.name)
                if validation_error:
                    parse_error = validation_error

            # Extract criterion scores if present
            criterion_scores = score_data.get("criterion_scores", {})

            # Extract feedback if present
            feedback = score_data.get("feedback", "")

        elif isinstance(score_data, (int, float)):
            # Handle case where LLM just returns array of numbers
            overall_score, validation_error = self._validate_score(score_data, feature.name)
            if validation_error:
                parse_error = validation_error
        else:
            parse_error = f"Unexpected score format: {type(score_data).__name__}"
            logger.warning(f"[{index}] {feature.name}: {parse_error}")
            if self.strict_mode:
                raise ScoreParseError(feature.name, parse_error, score_data)

        return FeatureScore(
            feature_name=feature.name,
            criterion_scores=criterion_scores,
            overall_score=overall_score,
            feedback=feedback,
            parse_error=parse_error
        )

    def _log_evaluation_summary(
        self,
        scores: List[FeatureScore],
        iteration: int,
        max_iterations: int,
        should_continue: bool
    ) -> Dict[str, Any]:
        """
        Log evaluation summary with structured data.

        Returns summary dict for programmatic access.
        """
        if not scores:
            logger.warning("No scores to summarize")
            return {"error": "no_scores"}

        avg_score = sum(fs.overall_score for fs in scores) / len(scores)
        parse_errors = sum(1 for fs in scores if fs.parse_error is not None)

        summary = {
            "iteration": iteration + 1,
            "max_iterations": max_iterations,
            "features_evaluated": len(scores),
            "average_score": round(avg_score, 2),
            "threshold": self.score_threshold,
            "parse_errors": parse_errors,
            "should_continue": should_continue,
        }

        # Determine reason for decision
        if not should_continue:
            if avg_score >= self.score_threshold:
                summary["stop_reason"] = f"avg_score {avg_score:.2f} >= threshold {self.score_threshold}"
            else:
                summary["stop_reason"] = f"reached max iterations ({max_iterations})"
        else:
            summary["continue_reason"] = f"avg_score {avg_score:.2f} < threshold {self.score_threshold}"

        # Log structured summary
        logger.info("Evaluation summary", extra=summary)

        # Also print human-readable summary
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - Iteration {summary['iteration']}/{max_iterations}")
        print(f"{'='*60}")
        print(f"Features evaluated: {summary['features_evaluated']}")
        print(f"Average score: {summary['average_score']:.2f} (threshold: {self.score_threshold})")
        if parse_errors > 0:
            print(f"⚠️  Parse errors: {parse_errors} (scores may be unreliable)")
        print(f"Decision: {'CONTINUE iterating' if should_continue else 'STOP'}")
        if "stop_reason" in summary:
            print(f"Reason: {summary['stop_reason']}")
        elif "continue_reason" in summary:
            print(f"Reason: {summary['continue_reason']}")
        print(f"{'='*60}\n")

        return summary

    def _validate_inputs(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: Optional[EvaluationRubric] = None
    ) -> None:
        """
        Validate inputs before evaluation.

        Raises:
            EvaluationError: If inputs are invalid
        """
        if not business_problem or not business_problem.strip():
            raise EvaluationError("Business problem cannot be empty")

        if not features:
            raise EvaluationError("Features list cannot be empty")

        if rubric is not None:
            if not rubric.criteria:
                raise EvaluationError("Rubric must have at least one criterion")

            # Validate weights sum approximately to 1.0
            total_weight = sum(c.weight for c in rubric.criteria)
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Rubric weights sum to {total_weight}, expected 1.0")

    def _parse_evaluation_result(self, result: dict, features: List[Feature]) -> FeatureEvaluation:
        """
        Parse LLM result into FeatureEvaluation with proper error tracking.

        Args:
            result: Raw dict from LLM
            features: List of features being evaluated (for fallback scores)

        Returns:
            FeatureEvaluation object

        Raises:
            EvaluationError: In strict mode, if parsing fails completely
        """
        feature_scores = []
        raw_scores = result.get("feature_scores", [])

        if not raw_scores:
            error_msg = "LLM returned no feature_scores"
            logger.error(error_msg)
            if self.strict_mode:
                raise EvaluationError(error_msg)
            # Create error scores for all features
            return FeatureEvaluation(
                feature_scores=[
                    FeatureScore(
                        feature_name=f.name,
                        criterion_scores={},
                        overall_score=5.0,
                        feedback="",
                        parse_error="No scores returned by LLM"
                    )
                    for f in features
                ],
                improvement_suggested=result.get("improvement_suggested", False),
                improvement_recommendations=result.get("improvement_recommendations")
            )

        # Use positional matching - LLM returns scores in same order as features
        for i, feature in enumerate(features):
            if i < len(raw_scores):
                score = self._parse_single_score(raw_scores[i], feature, i)
                feature_scores.append(score)
            else:
                # No score for this feature
                logger.warning(f"No score at index {i} for feature '{feature.name}'")
                if self.strict_mode:
                    raise ScoreParseError(feature.name, "Missing from LLM response")
                feature_scores.append(FeatureScore(
                    feature_name=feature.name,
                    criterion_scores={},
                    overall_score=5.0,
                    feedback="",
                    parse_error="No score returned by LLM"
                ))

        return FeatureEvaluation(
            feature_scores=feature_scores,
            improvement_suggested=result.get("improvement_suggested", False),
            improvement_recommendations=result.get("improvement_recommendations")
        )

    def _parse_compact_evaluation(self, result: dict, features: List[Feature]) -> FeatureEvaluation:
        """
        Parse compact evaluation result with proper error tracking.
        Uses positional matching since features are sent in order.

        Args:
            result: Raw dict from LLM
            features: List of features being evaluated (for fallback)

        Returns:
            FeatureEvaluation object
        """
        feature_scores = []
        raw_scores = result.get("scores", [])

        # Handle case where scores might be nested differently
        if not raw_scores and "feature_scores" in result:
            raw_scores = result.get("feature_scores", [])

        logger.debug(f"Received {len(raw_scores)} scores from LLM for {len(features)} features")

        if not raw_scores:
            error_msg = "LLM returned no scores"
            logger.error(error_msg)
            if self.strict_mode:
                raise EvaluationError(error_msg)

        # Use POSITIONAL matching - LLM returns scores in the same order as features
        for i, feature in enumerate(features):
            if i < len(raw_scores):
                score = self._parse_single_score(raw_scores[i], feature, i)
                feature_scores.append(score)
            else:
                # No score for this feature
                logger.warning(f"No score at index {i} for feature '{feature.name}'")
                if self.strict_mode:
                    raise ScoreParseError(feature.name, "Missing from LLM response")
                feature_scores.append(FeatureScore(
                    feature_name=feature.name,
                    criterion_scores={},
                    overall_score=5.0,
                    feedback="",
                    parse_error="No score returned by LLM"
                ))

        # Extract continue flag - try multiple possible field names
        continue_flag = result.get("continue", result.get("continue_", False))
        if isinstance(continue_flag, str):
            continue_flag = continue_flag.lower() in ("true", "yes", "1")

        feedback = result.get("feedback", result.get("improvement_recommendations"))

        avg_score = sum(fs.overall_score for fs in feature_scores) / len(feature_scores) if feature_scores else 0
        logger.debug(f"Average score: {avg_score:.2f}, continue flag: {continue_flag}")

        return FeatureEvaluation(
            feature_scores=feature_scores,
            improvement_suggested=bool(continue_flag),
            improvement_recommendations=feedback
        )

    def create_rubric(self, business_problem: str) -> EvaluationRubric:
        """
        Create an evaluation rubric for the business problem.

        Args:
            business_problem: Natural language description of the problem

        Returns:
            EvaluationRubric object
        """
        if self.compact_mode:
            return self.default_rubric

        chain = self.rubric_prompt | self.llm | self.rubric_parser
        result = chain.invoke({"business_problem": business_problem})
        return EvaluationRubric(**result)

    def _evaluate_batch(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: Optional[EvaluationRubric],
        iteration: int,
        max_iterations: int
    ) -> List[FeatureScore]:
        """
        Evaluate a single batch of features.

        Args:
            business_problem: The business problem context
            features: Batch of features to evaluate
            rubric: The evaluation rubric (None for compact mode)
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            List of FeatureScore objects
        """
        if self.compact_mode:
            features_text = ", ".join([f.name for f in features])

            chain = self.evaluation_prompt | self.llm | self.evaluation_parser
            result = chain.invoke({
                "business_problem": business_problem,
                "features": features_text,
                "iteration": iteration + 1,
                "max_iterations": max_iterations
            })

            # Parse scores from this batch
            raw_scores = result.get("scores", [])
            if not raw_scores and "feature_scores" in result:
                raw_scores = result.get("feature_scores", [])

            batch_scores = []
            for i, feature in enumerate(features):
                if i < len(raw_scores):
                    score = self._parse_single_score(raw_scores[i], feature, i)
                    batch_scores.append(score)
                else:
                    if self.strict_mode:
                        raise ScoreParseError(feature.name, "Missing from batch response")
                    batch_scores.append(FeatureScore(
                        feature_name=feature.name,
                        criterion_scores={},
                        overall_score=5.0,
                        feedback="",
                        parse_error="No score returned in batch"
                    ))

            return batch_scores
        else:
            # Full mode
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

            chain = self.evaluation_prompt | self.llm | self.evaluation_parser
            result = chain.invoke({
                "business_problem": business_problem,
                "rubric": rubric_text,
                "features": features_text,
                "iteration": iteration + 1,
                "max_iterations": max_iterations
            })

            # Parse scores using positional matching
            batch_scores = []
            raw_scores = result.get("feature_scores", [])

            for i, feature in enumerate(features):
                if i < len(raw_scores):
                    score = self._parse_single_score(raw_scores[i], feature, i)
                    batch_scores.append(score)
                else:
                    if self.strict_mode:
                        raise ScoreParseError(feature.name, "Missing from batch response")
                    batch_scores.append(FeatureScore(
                        feature_name=feature.name,
                        criterion_scores={},
                        overall_score=5.0,
                        feedback="",
                        parse_error="No score returned in batch"
                    ))

            return batch_scores

    def evaluate_features(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: EvaluationRubric,
        iteration: int,
        max_iterations: int,
        previous_scores: Optional[Dict[str, FeatureScore]] = None
    ) -> FeatureEvaluation:
        """
        Evaluate features against the rubric.

        Args:
            business_problem: The business problem context
            features: List of features to evaluate
            rubric: The evaluation rubric
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            previous_scores: Optional dict of feature_name -> FeatureScore for features
                           that don't need re-evaluation (token optimization)

        Returns:
            FeatureEvaluation with scores and recommendations
        """
        # Validate inputs
        self._validate_inputs(business_problem, features, rubric)

        # Separate features that need evaluation vs those with existing scores
        features_to_evaluate = []
        carried_scores = []

        if previous_scores:
            for feature in features:
                if feature.name in previous_scores:
                    carried_scores.append(previous_scores[feature.name])
                    logger.debug(f"Carrying forward score for '{feature.name}'")
                else:
                    features_to_evaluate.append(feature)
            logger.info(f"Evaluating {len(features_to_evaluate)} new features, carrying {len(carried_scores)} existing scores")
        else:
            features_to_evaluate = features

        # If no features need evaluation, return carried scores
        if not features_to_evaluate:
            avg_score = sum(fs.overall_score for fs in carried_scores) / len(carried_scores)
            should_continue = avg_score < self.score_threshold and iteration < max_iterations - 1
            self._log_evaluation_summary(carried_scores, iteration, max_iterations, should_continue)
            return FeatureEvaluation(
                feature_scores=carried_scores,
                improvement_suggested=should_continue,
                improvement_recommendations="All features carried from previous iteration" if not should_continue else None
            )

        all_scores = []

        # Batch evaluation if needed
        if len(features_to_evaluate) > self.batch_size:
            logger.info(f"Evaluating {len(features_to_evaluate)} features in batches of {self.batch_size}")

            for batch_start in range(0, len(features_to_evaluate), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(features_to_evaluate))
                batch = features_to_evaluate[batch_start:batch_end]
                batch_num = batch_start // self.batch_size + 1

                logger.debug(f"Evaluating batch {batch_num}: features {batch_start+1}-{batch_end}")

                batch_scores = self._evaluate_batch(
                    business_problem, batch, rubric, iteration, max_iterations
                )
                all_scores.extend(batch_scores)
                logger.debug(f"Batch {batch_num} returned {len(batch_scores)} scores")

        else:
            # Single batch - evaluate all at once
            if self.compact_mode:
                features_text = ", ".join([f.name for f in features_to_evaluate])

                chain = self.evaluation_prompt | self.llm | self.evaluation_parser
                result = chain.invoke({
                    "business_problem": business_problem,
                    "features": features_text,
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations
                })

                evaluation = self._parse_compact_evaluation(result, features_to_evaluate)
                all_scores = evaluation.feature_scores
            else:
                features_text = "\n\n".join([
                    f"Feature: {f.name}\n"
                    f"Description: {f.description}\n"
                    f"Data Type: {f.data_type}\n"
                    f"Taxonomy: {f.taxonomy}\n"
                    f"Rationale: {f.rationale}"
                    for f in features_to_evaluate
                ])

                rubric_text = f"Criteria:\n" + "\n".join([
                    f"- {c.name} (weight: {c.weight}): {c.description}"
                    for c in rubric.criteria
                ])

                chain = self.evaluation_prompt | self.llm | self.evaluation_parser
                result = chain.invoke({
                    "business_problem": business_problem,
                    "rubric": rubric_text,
                    "features": features_text,
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations
                })

                evaluation = self._parse_evaluation_result(result, features_to_evaluate)
                all_scores = evaluation.feature_scores

        # Combine carried scores with new scores
        combined_scores = carried_scores + all_scores

        # Calculate decision based on all scores
        avg_score = sum(fs.overall_score for fs in combined_scores) / len(combined_scores)
        should_continue = avg_score < self.score_threshold and iteration < max_iterations - 1

        # Log summary
        self._log_evaluation_summary(combined_scores, iteration, max_iterations, should_continue)

        return FeatureEvaluation(
            feature_scores=combined_scores,
            improvement_suggested=should_continue,
            improvement_recommendations="Continue improving low-scoring features" if should_continue else None
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function.

        Args:
            state: Current workflow state

        Returns:
            Updated state with evaluation results
        """
        business_problem = state["business_problem"]
        features = state["current_features"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]

        # Validate we have features to evaluate
        if not features:
            raise EvaluationError("No features to evaluate in state")

        # Create rubric on first iteration
        if state.get("rubric") is None:
            rubric = self.create_rubric(business_problem)
        else:
            rubric = state["rubric"]

        # Build previous scores dict for token optimization
        # Only carry forward scores for kept features (they don't need re-evaluation)
        previous_scores = None
        kept_features = state.get("kept_features", [])
        if kept_features and state.get("evaluations"):
            previous_scores = {}
            kept_names = {f.name for f in kept_features}
            for score in state["evaluations"]:
                if score.feature_name in kept_names:
                    previous_scores[score.feature_name] = score
            logger.info(f"Carrying forward {len(previous_scores)} scores for kept features")

        # Evaluate features
        evaluation = self.evaluate_features(
            business_problem,
            features,
            rubric,
            iteration,
            max_iterations,
            previous_scores=previous_scores
        )

        # Check for parse errors and warn
        parse_error_count = sum(1 for s in evaluation.feature_scores if s.parse_error)
        if parse_error_count > 0:
            logger.warning(f"{parse_error_count} features had parse errors - scores may be unreliable")

        # Separate high-scoring and low-scoring features
        kept_features = []
        low_scoring_features = []

        for i, score in enumerate(evaluation.feature_scores):
            if i < len(features):
                if score.overall_score >= self.score_threshold:
                    kept_features.append(features[i])
                else:
                    low_scoring_features.append(features[i])

        low_scoring_count = len(low_scoring_features)

        # Log the separation
        print(f"\n--- Feature Scoring Summary ---")
        print(f"Features above threshold ({self.score_threshold}): {len(kept_features)}")
        print(f"Features below threshold: {low_scoring_count}")
        if parse_error_count > 0:
            print(f"⚠️  Features with parse errors: {parse_error_count}")
        if kept_features:
            print(f"Kept features: {[f.name for f in kept_features]}")
        if low_scoring_features:
            print(f"To regenerate: {[f.name for f in low_scoring_features]}")
        print(f"-------------------------------\n")

        # Determine if we should continue iterating
        # Continue if there are low-scoring features AND we haven't reached max iterations
        should_continue = (
            low_scoring_count > 0 and
            iteration < max_iterations - 1
        )

        # Build improvement recommendations from low-scoring feature feedback
        # Include data type distribution to maintain variety
        low_scoring_type_counts = {}
        if should_continue and low_scoring_features:
            # Count data types of low-scoring features
            for f in low_scoring_features:
                dtype = str(f.data_type.value) if hasattr(f.data_type, 'value') else str(f.data_type)
                low_scoring_type_counts[dtype] = low_scoring_type_counts.get(dtype, 0) + 1

            # Build detailed recommendations with type distribution
            type_info = ", ".join([f"{count} {dtype}" for dtype, count in low_scoring_type_counts.items()])
            recommendations = f"Replace {low_scoring_count} low-scoring features ({type_info}). "
            recommendations += f"Maintain data type variety: generate replacements with similar type distribution. "
            if evaluation.improvement_recommendations:
                recommendations += evaluation.improvement_recommendations

            logger.info(f"Low-scoring feature types: {low_scoring_type_counts}")
        else:
            recommendations = evaluation.improvement_recommendations

        return {
            "rubric": rubric,
            "evaluations": evaluation.feature_scores,
            "improvement_recommendations": recommendations,
            "kept_features": kept_features,
            "low_scoring_count": low_scoring_count,
            "low_scoring_type_counts": low_scoring_type_counts if low_scoring_type_counts else None,
            "converged": not should_continue,
            "iteration": iteration + 1
        }
