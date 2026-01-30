"""
Evaluator Agent - Creates rubrics and evaluates generated features.
"""
import json
import re
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain_core.exceptions import OutputParserException
from ..models.schemas import (
    Feature,
    EvaluationRubric,
    RubricCriterion,
    FeatureEvaluation,
    FeatureScore,
    CompactFeatureEvaluation,
    CompactFeatureScore
)


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

    def __init__(self, llm, compact_mode: bool = False):
        """
        Initialize the Evaluator Agent.

        Args:
            llm: Language model instance (OpenAI or Anthropic)
            compact_mode: If True, use minimal schema for faster evaluation
        """
        self.llm = llm
        self.compact_mode = compact_mode

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

    def _parse_evaluation_result(self, result: dict, features: List[Feature]) -> FeatureEvaluation:
        """
        Parse LLM result into FeatureEvaluation with fallbacks for malformed responses.

        Args:
            result: Raw dict from LLM
            features: List of features being evaluated (for fallback scores)

        Returns:
            FeatureEvaluation object
        """
        try:
            return FeatureEvaluation(**result)
        except Exception as e:
            # Try to salvage what we can
            print(f"Warning: Evaluation parsing failed ({e}), attempting recovery...")

            feature_scores = []

            # Try to extract feature_scores
            raw_scores = result.get("feature_scores", [])
            for i, score_data in enumerate(raw_scores):
                try:
                    if isinstance(score_data, dict):
                        # Ensure required fields exist
                        score_data.setdefault("feature_name", features[i].name if i < len(features) else f"feature_{i}")
                        score_data.setdefault("criterion_scores", {})
                        score_data.setdefault("overall_score", 5.0)
                        score_data.setdefault("feedback", "")
                        feature_scores.append(FeatureScore(**score_data))
                except Exception:
                    continue

            # If no scores extracted, create default scores for all features
            if not feature_scores:
                feature_scores = [
                    FeatureScore(
                        feature_name=f.name,
                        criterion_scores={},
                        overall_score=5.0,
                        feedback="Unable to parse evaluation"
                    )
                    for f in features
                ]

            return FeatureEvaluation(
                feature_scores=feature_scores,
                improvement_suggested=result.get("improvement_suggested", False),
                improvement_recommendations=result.get("improvement_recommendations")
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

    def _parse_compact_evaluation(self, result: dict, features: List[Feature]) -> FeatureEvaluation:
        """
        Parse compact evaluation result with robust fallbacks.
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

        # Debug: show what we got
        print(f"Debug: Received {len(raw_scores)} scores from LLM for {len(features)} features")

        # Use POSITIONAL matching - LLM returns scores in the same order as features
        # This is more reliable than name matching which can fail due to slight differences
        for i, feature in enumerate(features):
            if i < len(raw_scores):
                score_data = raw_scores[i]
                try:
                    if isinstance(score_data, dict):
                        # Try different field names for score
                        score_val = score_data.get("score", score_data.get("overall_score", 5.0))

                        # Ensure score is a valid number
                        try:
                            score_float = float(score_val)
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid score value at index {i}: {score_val}, using default")
                            score_float = 5.0

                        feature_scores.append(FeatureScore(
                            feature_name=feature.name,  # Use actual feature name
                            criterion_scores={},
                            overall_score=score_float,
                            feedback=""
                        ))
                    elif isinstance(score_data, (int, float)):
                        # Handle case where LLM just returns array of numbers
                        feature_scores.append(FeatureScore(
                            feature_name=feature.name,
                            criterion_scores={},
                            overall_score=float(score_data),
                            feedback=""
                        ))
                    else:
                        print(f"Warning: Unexpected score format at index {i}: {type(score_data)}")
                        feature_scores.append(FeatureScore(
                            feature_name=feature.name,
                            criterion_scores={},
                            overall_score=5.0,
                            feedback="Could not parse score"
                        ))
                except Exception as e:
                    print(f"Warning: Failed to parse score at index {i}: {e}")
                    feature_scores.append(FeatureScore(
                        feature_name=feature.name,
                        criterion_scores={},
                        overall_score=5.0,
                        feedback="Parse error"
                    ))
            else:
                # No score for this feature, use default
                print(f"Warning: No score at index {i} for feature '{feature.name}'")
                feature_scores.append(FeatureScore(
                    feature_name=feature.name,
                    criterion_scores={},
                    overall_score=5.0,
                    feedback="No score returned by LLM"
                ))

        # Extract continue flag - try multiple possible field names
        continue_flag = result.get("continue", result.get("continue_", False))
        if isinstance(continue_flag, str):
            continue_flag = continue_flag.lower() in ("true", "yes", "1")

        feedback = result.get("feedback", result.get("improvement_recommendations"))

        avg_score = sum(fs.overall_score for fs in feature_scores) / len(feature_scores) if feature_scores else 0
        print(f"Debug: Average score: {avg_score:.2f}, continue: {continue_flag}")

        return FeatureEvaluation(
            feature_scores=feature_scores,
            improvement_suggested=bool(continue_flag),
            improvement_recommendations=feedback
        )

    def evaluate_features(
        self,
        business_problem: str,
        features: List[Feature],
        rubric: EvaluationRubric,
        iteration: int,
        max_iterations: int
    ) -> FeatureEvaluation:
        """
        Evaluate features against the rubric.

        Args:
            business_problem: The business problem context
            features: List of features to evaluate
            rubric: The evaluation rubric
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            FeatureEvaluation with scores and recommendations
        """

        if self.compact_mode:
            # Compact feature list - number them for clarity
            features_text = ", ".join([f.name for f in features])

            chain = self.evaluation_prompt | self.llm | self.evaluation_parser
            result = chain.invoke({
                "business_problem": business_problem,
                "features": features_text,
                "iteration": iteration + 1,
                "max_iterations": max_iterations
            })

            # Debug: show raw result
            print(f"Debug: Raw LLM result keys: {result.keys() if isinstance(result, dict) else type(result)}")
            if isinstance(result, dict):
                scores = result.get("scores", [])
                print(f"Debug: scores field has {len(scores)} items")
                if scores and len(scores) > 0:
                    print(f"Debug: First score entry: {scores[0]}")

            return self._parse_compact_evaluation(result, features)
        else:
            # Full feature details
            features_text = "\n\n".join([
                f"Feature: {f.name}\n"
                f"Description: {f.description}\n"
                f"Data Type: {f.data_type}\n"
                f"Taxonomy: {f.taxonomy}\n"
                f"Rationale: {f.rationale}"
                for f in features
            ])

            # Format rubric
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

            return self._parse_evaluation_result(result, features)

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

        # Create rubric on first iteration
        if state.get("rubric") is None:
            rubric = self.create_rubric(business_problem)
        else:
            rubric = state["rubric"]

        # Evaluate features
        evaluation = self.evaluate_features(
            business_problem,
            features,
            rubric,
            iteration,
            max_iterations
        )

        # Determine if we should continue iterating
        should_continue = (
            evaluation.improvement_suggested and
            iteration < max_iterations - 1
        )

        return {
            "rubric": rubric,
            "evaluations": evaluation.feature_scores,
            "improvement_recommendations": evaluation.improvement_recommendations,
            "converged": not should_continue,
            "iteration": iteration + 1
        }
