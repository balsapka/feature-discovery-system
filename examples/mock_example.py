"""
Mock example that simulates the Feature Discovery workflow without requiring an API key.
Uses fake LLM responses to demonstrate the workflow structure.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any, List
from src.models.schemas import (
    Feature,
    DataType,
    EvaluationRubric,
    RubricCriterion,
    FeatureScore,
    FeatureEvaluation,
    GeneratorOutput
)


class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self):
        self.call_count = 0

    def invoke(self, prompt) -> str:
        """Return mock response based on call count."""
        self.call_count += 1
        return f"Mock response {self.call_count}"


class MockFeatureGenerator:
    """Mock feature generator with predefined outputs."""

    def __init__(self):
        self.iteration = 0

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.iteration = state.get("iteration", 0)

        if self.iteration == 0:
            features = self._get_initial_features()
        else:
            features = self._get_improved_features()

        return {
            "current_features": features,
            "taxonomy_explanation": "Features organized by: Customer Behavior, Account Metrics, External Factors"
        }

    def _get_initial_features(self) -> List[Feature]:
        return [
            Feature(
                name="account_balance_avg_30d",
                description="Average account balance over the last 30 days",
                data_type=DataType.TIME_SERIES,
                taxonomy="Account Metrics/Balance",
                rationale="Balance trends indicate financial stability and engagement"
            ),
            Feature(
                name="transaction_count_30d",
                description="Number of transactions in the last 30 days",
                data_type=DataType.TIME_SERIES,
                taxonomy="Customer Behavior/Transactions",
                rationale="Transaction frequency indicates account activity and engagement"
            ),
            Feature(
                name="customer_tenure_months",
                description="Number of months since account opening",
                data_type=DataType.STRUCTURED,
                taxonomy="Account Metrics/Demographics",
                rationale="Longer tenure often correlates with lower churn risk"
            ),
            Feature(
                name="support_tickets_90d",
                description="Number of customer support interactions in 90 days",
                data_type=DataType.STRUCTURED,
                taxonomy="Customer Behavior/Support",
                rationale="High support volume may indicate dissatisfaction"
            ),
        ]

    def _get_improved_features(self) -> List[Feature]:
        """Return improved features based on feedback."""
        initial = self._get_initial_features()
        initial.extend([
            Feature(
                name="competitor_rate_differential",
                description="Difference between our rates and top competitor rates",
                data_type=DataType.EXTERNAL,
                taxonomy="External Factors/Competition",
                rationale="Rate competitiveness influences customer retention decisions"
            ),
            Feature(
                name="mobile_app_logins_30d",
                description="Number of mobile app logins in last 30 days",
                data_type=DataType.TIME_SERIES,
                taxonomy="Customer Behavior/Digital Engagement",
                rationale="Digital engagement indicates modern banking adoption"
            ),
        ])
        return initial


class MockEvaluator:
    """Mock evaluator with predefined outputs."""

    def __init__(self):
        self.rubric = None

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        iteration = state.get("iteration", 0)
        features = state.get("current_features", [])

        # Create rubric on first call
        if self.rubric is None:
            self.rubric = self._create_rubric()

        # Evaluate features
        scores = self._evaluate_features(features, iteration)

        # Decide if we should continue
        should_continue = iteration < 1  # Only do 2 iterations in mock

        return {
            "rubric": self.rubric,
            "evaluations": scores,
            "improvement_recommendations": "Add external data sources and digital engagement metrics" if should_continue else None,
            "converged": not should_continue,
            "iteration": iteration + 1
        }

    def _create_rubric(self) -> EvaluationRubric:
        return EvaluationRubric(
            criteria=[
                RubricCriterion(
                    name="Relevance",
                    description="How directly the feature relates to churn prediction",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Data Availability",
                    description="Likelihood the data exists in banking systems",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Predictive Power",
                    description="Expected impact on model performance",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Interpretability",
                    description="How easily stakeholders can understand the feature",
                    weight=0.2
                ),
            ],
            rationale="Rubric focused on practical implementation and business value for churn prediction"
        )

    def _evaluate_features(self, features: List[Feature], iteration: int) -> List[FeatureScore]:
        scores = []
        base_score = 7.0 if iteration == 0 else 8.0  # Scores improve with iteration

        for i, feature in enumerate(features):
            score = FeatureScore(
                feature_name=feature.name,
                criterion_scores={
                    "Relevance": base_score + (i % 2),
                    "Data Availability": base_score + 0.5,
                    "Predictive Power": base_score - 0.5 + (i % 3) * 0.5,
                    "Interpretability": base_score + 1
                },
                overall_score=base_score + 0.3 * i,
                feedback=f"Good feature. Consider adding time-based variants." if iteration == 0 else "Well-defined feature with clear business value."
            )
            scores.append(score)

        return scores


def run_mock_workflow():
    """Run the mock workflow to demonstrate the system."""

    print("=" * 80)
    print("MOCK FEATURE DISCOVERY WORKFLOW")
    print("(No API key required - using simulated responses)")
    print("=" * 80)

    business_problem = """
    Predict customer churn in retail banking. We want to identify customers
    likely to close their accounts in the next 90 days.
    """

    print(f"\nBusiness Problem: {business_problem.strip()}\n")

    # Initialize mock agents
    generator = MockFeatureGenerator()
    evaluator = MockEvaluator()

    # Initialize state
    state = {
        "business_problem": business_problem,
        "current_features": None,
        "rubric": None,
        "evaluations": None,
        "iteration": 0,
        "max_iterations": 3,
        "converged": False,
    }

    iteration = 0
    while not state["converged"] and iteration < state["max_iterations"]:
        print(f"\n{'─' * 40}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'─' * 40}")

        # Generate features
        print("\n📝 Generating features...")
        gen_result = generator(state)
        state.update(gen_result)

        print(f"   Generated {len(state['current_features'])} features:")
        for f in state['current_features']:
            print(f"   • {f.name} ({f.data_type.value})")

        # Evaluate features
        print("\n📊 Evaluating features...")
        eval_result = evaluator(state)
        state.update(eval_result)

        print(f"   Rubric criteria: {[c.name for c in state['rubric'].criteria]}")

        avg_score = sum(e.overall_score for e in state['evaluations']) / len(state['evaluations'])
        print(f"   Average score: {avg_score:.2f}/10")

        if state.get("improvement_recommendations"):
            print(f"   💡 Recommendations: {state['improvement_recommendations']}")

        iteration = state["iteration"]

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\nTotal Iterations: {state['iteration']}")
    print(f"Converged: {state['converged']}")

    print("\n📋 Evaluation Rubric:")
    for c in state['rubric'].criteria:
        print(f"   • {c.name} (weight: {c.weight}): {c.description}")

    print("\n🎯 Final Features with Scores:")
    for feature, score in zip(state['current_features'], state['evaluations']):
        print(f"\n   {feature.name}")
        print(f"      Type: {feature.data_type.value}")
        print(f"      Score: {score.overall_score:.1f}/10")
        print(f"      Feedback: {score.feedback}")

    print("\n✅ Mock workflow completed successfully!")
    print("   This demonstrates the workflow structure without API calls.")
    print("   To run with real LLM, set ANTHROPIC_API_KEY and use basic_example.py")


if __name__ == "__main__":
    run_mock_workflow()
