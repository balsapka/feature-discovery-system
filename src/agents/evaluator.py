"""
Evaluator Agent - Creates rubrics and evaluates generated features.
"""
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..models.schemas import (
    Feature,
    EvaluationRubric,
    RubricCriterion,
    FeatureEvaluation,
    FeatureScore,
    CompactFeatureEvaluation,
    CompactFeatureScore
)


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
            self.evaluation_parser = JsonOutputParser(pydantic_object=CompactFeatureEvaluation)
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
            # Compact feature list
            features_text = ", ".join([f.name for f in features])

            chain = self.evaluation_prompt | self.llm | self.evaluation_parser
            result = chain.invoke({
                "business_problem": business_problem,
                "features": features_text,
                "iteration": iteration + 1,
                "max_iterations": max_iterations
            })

            # Convert compact result to full format
            compact_eval = CompactFeatureEvaluation(**result)
            feature_scores = [s.to_feature_score() for s in compact_eval.scores]
            return FeatureEvaluation(
                feature_scores=feature_scores,
                improvement_suggested=compact_eval.continue_,
                improvement_recommendations=compact_eval.feedback
            )
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

            return FeatureEvaluation(**result)

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
