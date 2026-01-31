"""
Evaluator Agent - Creates rubrics and evaluates generated features.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class EvaluatorAgent:
    """
    Agent responsible for creating evaluation rubrics and scoring features.
    Uses an evaluator-optimizer pattern to iteratively improve feature quality.
    """

    def __init__(
        self,
        llm,
        compact_mode: bool = False,
        parallel: bool = False,
        score_threshold: float = 7.0,
        batch_size: int = 10
    ):
        """
        Initialize the Evaluator Agent.

        Args:
            llm: Language model instance (OpenAI or Anthropic)
            compact_mode: If True, use minimal schema for faster evaluation
            parallel: If True, evaluate batches in parallel for faster execution
            score_threshold: Minimum average score to stop iterating (default 7.0)
            batch_size: Number of features per evaluation batch (default 10)
        """
        self.llm = llm
        self.compact_mode = compact_mode
        self.parallel = parallel
        self.score_threshold = score_threshold
        self.batch_size = batch_size

        if compact_mode:
            self.evaluation_parser = RobustJsonOutputParser()
            self.evaluation_prompt = self._create_compact_evaluation_prompt()
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
    "scores": [{{"name": "feature_name", "score": 7.5, "feedback": "brief feedback"}}],
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

    def _try_parse_score(self, score_data: Any, feature_name: str) -> Optional[FeatureScore]:
        """
        Try to parse a score, returning None if it can't be validated.

        Features that fail parsing are simply excluded rather than using fallback values.
        """
        try:
            if isinstance(score_data, dict):
                # Try different field names for score
                score_val = score_data.get("score") or score_data.get("overall_score")
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
                    criterion_scores=score_data.get("criterion_scores", {}),
                    overall_score=score_float,
                    feedback=score_data.get("feedback", "")
                )
            elif isinstance(score_data, (int, float)):
                score_float = float(score_data)
                if score_float < 0 or score_float > 10:
                    return None
                return FeatureScore(
                    feature_name=feature_name,
                    criterion_scores={},
                    overall_score=score_float,
                    feedback=""
                )
            else:
                logger.debug(f"Unexpected score format for '{feature_name}': {type(score_data)}")
                return None
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse score for '{feature_name}': {e}")
            return None

    def _log_evaluation_summary(
        self,
        scores: List[FeatureScore],
        iteration: int,
        max_iterations: int,
        should_continue: bool,
        skipped_count: int = 0
    ) -> None:
        """Log evaluation summary."""
        if not scores:
            logger.warning("No valid scores to summarize")
            return

        avg_score = sum(fs.overall_score for fs in scores) / len(scores)

        logger.info("Evaluation summary", extra={
            "iteration": iteration + 1,
            "max_iterations": max_iterations,
            "features_scored": len(scores),
            "features_skipped": skipped_count,
            "average_score": round(avg_score, 2),
            "should_continue": should_continue,
        })

        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - Iteration {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")
        print(f"Features scored: {len(scores)}")
        if skipped_count > 0:
            print(f"Features skipped (invalid scores): {skipped_count}")
        print(f"Average score: {avg_score:.2f} (threshold: {self.score_threshold})")
        print(f"Decision: {'CONTINUE iterating' if should_continue else 'STOP'}")
        print(f"{'='*60}\n")

    def create_rubric(self, business_problem: str) -> EvaluationRubric:
        """Create an evaluation rubric for the business problem."""
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
    ) -> Tuple[List[FeatureScore], int]:
        """
        Evaluate a batch of features.

        Returns:
            Tuple of (valid_scores, skipped_count)
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
            raw_scores = result.get("scores", result.get("feature_scores", []))
        else:
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
            raw_scores = result.get("feature_scores", [])

        # Parse scores, filtering out invalid ones
        valid_scores = []
        skipped = 0

        for i, feature in enumerate(features):
            if i < len(raw_scores):
                score = self._try_parse_score(raw_scores[i], feature.name)
                if score:
                    valid_scores.append(score)
                else:
                    skipped += 1
            else:
                # No score returned for this feature
                skipped += 1

        return valid_scores, skipped

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

        Features that fail to parse are excluded from results.
        """
        if not business_problem or not business_problem.strip():
            raise EvaluationError("Business problem cannot be empty")
        if not features:
            raise EvaluationError("Features list cannot be empty")

        # Separate features needing evaluation vs carried forward
        features_to_evaluate = []
        carried_scores = []

        if previous_scores:
            for feature in features:
                if feature.name in previous_scores:
                    carried_scores.append(previous_scores[feature.name])
                else:
                    features_to_evaluate.append(feature)
            logger.info(f"Evaluating {len(features_to_evaluate)} new features, carrying {len(carried_scores)} existing")
        else:
            features_to_evaluate = features

        all_scores = []
        total_skipped = 0

        if features_to_evaluate:
            # Batch if needed
            if len(features_to_evaluate) > self.batch_size:
                batches = [
                    features_to_evaluate[i:i + self.batch_size]
                    for i in range(0, len(features_to_evaluate), self.batch_size)
                ]
                logger.info(f"Evaluating {len(features_to_evaluate)} features in {len(batches)} batches of up to {self.batch_size}")

                if self.parallel and len(batches) > 1:
                    # Parallel evaluation
                    logger.info(f"Using parallel evaluation with {len(batches)} workers")
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
                                batch_scores, skipped = future.result()
                                all_scores.extend(batch_scores)
                                total_skipped += skipped
                            except Exception as e:
                                logger.warning(f"Batch {batch_idx} failed in parallel mode: {e}, will retry sequentially")
                                failed_batches.append(batch)

                    # Retry failed batches sequentially to ensure all features get evaluated
                    if failed_batches:
                        logger.info(f"Retrying {len(failed_batches)} failed batches sequentially")
                        for batch in failed_batches:
                            try:
                                batch_scores, skipped = self._evaluate_batch(
                                    business_problem, batch, rubric, iteration, max_iterations
                                )
                                all_scores.extend(batch_scores)
                                total_skipped += skipped
                            except Exception as e:
                                logger.error(f"Batch retry also failed: {e}")
                                total_skipped += len(batch)
                else:
                    # Sequential evaluation
                    for batch in batches:
                        batch_scores, skipped = self._evaluate_batch(
                            business_problem, batch, rubric, iteration, max_iterations
                        )
                        all_scores.extend(batch_scores)
                        total_skipped += skipped
            else:
                batch_scores, total_skipped = self._evaluate_batch(
                    business_problem, features_to_evaluate, rubric, iteration, max_iterations
                )
                all_scores.extend(batch_scores)

        # Combine with carried scores
        combined_scores = carried_scores + all_scores

        if not combined_scores:
            raise EvaluationError("No valid scores could be parsed from evaluation")

        # Calculate decision
        avg_score = sum(fs.overall_score for fs in combined_scores) / len(combined_scores)
        should_continue = avg_score < self.score_threshold and iteration < max_iterations - 1

        self._log_evaluation_summary(combined_scores, iteration, max_iterations, should_continue, total_skipped)

        return FeatureEvaluation(
            feature_scores=combined_scores,
            improvement_suggested=should_continue,
            improvement_recommendations="Continue improving low-scoring features" if should_continue else None
        )

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
        previous_scores = None
        kept_features = state.get("kept_features", [])
        if kept_features and state.get("evaluations"):
            previous_scores = {}
            kept_names = {f.name for f in kept_features}
            for score in state["evaluations"]:
                if score.feature_name in kept_names:
                    previous_scores[score.feature_name] = score

        # Evaluate
        evaluation = self.evaluate_features(
            business_problem, features, rubric, iteration, max_iterations,
            previous_scores=previous_scores
        )

        # Build a map of feature name -> score for matching
        score_map = {s.feature_name: s for s in evaluation.feature_scores}

        # Separate high-scoring and low-scoring features
        # Only include features that have valid scores
        kept_features = []
        low_scoring_features = []

        for feature in features:
            score = score_map.get(feature.name)
            if score:
                if score.overall_score >= self.score_threshold:
                    kept_features.append(feature)
                else:
                    low_scoring_features.append(feature)
            # Features without scores are simply excluded

        low_scoring_count = len(low_scoring_features)

        print(f"\n--- Feature Scoring Summary ---")
        print(f"Features above threshold ({self.score_threshold}): {len(kept_features)}")
        print(f"Features below threshold: {low_scoring_count}")
        if kept_features:
            print(f"Kept: {[f.name for f in kept_features]}")
        if low_scoring_features:
            print(f"To regenerate: {[f.name for f in low_scoring_features]}")
        print(f"-------------------------------\n")

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

        return {
            "rubric": rubric,
            "evaluations": evaluation.feature_scores,
            "improvement_recommendations": recommendations,
            "kept_features": kept_features,
            "low_scoring_count": low_scoring_count,
            "low_scoring_type_counts": low_scoring_type_counts or None,
            "converged": not should_continue,
            "iteration": iteration + 1
        }
