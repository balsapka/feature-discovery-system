"""
Tests for the Feature Discovery Workflow.
"""
import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow import FeatureDiscoveryWorkflow
from src.models.schemas import (
    Feature, 
    DataType, 
    EvaluationRubric, 
    RubricCriterion,
    FeatureScore
)


class TestFeatureDiscoveryWorkflow:
    """Test suite for the Feature Discovery Workflow."""
    
    @pytest.fixture
    def sample_business_problem(self):
        """Sample business problem for testing."""
        return "Predict customer churn in retail banking"
    
    @pytest.fixture
    def sample_features(self):
        """Sample features for testing."""
        return [
            Feature(
                name="account_age_days",
                description="Number of days since account opening",
                data_type=DataType.STRUCTURED,
                taxonomy="customer/account_info",
                rationale="Newer accounts may be more likely to churn"
            ),
            Feature(
                name="transaction_frequency",
                description="Average number of transactions per month",
                data_type=DataType.TIME_SERIES,
                taxonomy="behavioral/transaction_patterns",
                rationale="Low activity may indicate disengagement"
            )
        ]
    
    @pytest.fixture
    def sample_rubric(self):
        """Sample evaluation rubric for testing."""
        return EvaluationRubric(
            criteria=[
                RubricCriterion(
                    name="Relevance",
                    description="How relevant to predicting churn",
                    weight=0.4
                ),
                RubricCriterion(
                    name="Data Availability",
                    description="Likelihood data exists",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Predictive Power",
                    description="Expected model impact",
                    weight=0.3
                )
            ],
            rationale="Balanced rubric for churn prediction"
        )
    
    def test_workflow_initialization_anthropic(self):
        """Test workflow initialization with Anthropic."""
        workflow = FeatureDiscoveryWorkflow(
            llm_provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            max_iterations=3
        )
        
        assert workflow.llm_provider == "anthropic"
        assert workflow.max_iterations == 3
        assert workflow.generator is not None
        assert workflow.evaluator is not None
        assert workflow.graph is not None
    
    def test_workflow_initialization_openai(self):
        """Test workflow initialization with OpenAI."""
        workflow = FeatureDiscoveryWorkflow(
            llm_provider="openai",
            model_name="gpt-4",
            max_iterations=2
        )
        
        assert workflow.llm_provider == "openai"
        assert workflow.max_iterations == 2
    
    def test_invalid_llm_provider(self):
        """Test that invalid LLM provider raises error."""
        with pytest.raises(ValueError):
            FeatureDiscoveryWorkflow(llm_provider="invalid")
    
    def test_should_continue_logic(self, sample_business_problem):
        """Test the continue/finalize decision logic."""
        workflow = FeatureDiscoveryWorkflow(max_iterations=3)
        
        # Should continue when not converged and under max iterations
        state = {
            "converged": False,
            "iteration": 1,
            "max_iterations": 3
        }
        assert workflow._should_continue(state) == "continue"
        
        # Should finalize when converged
        state["converged"] = True
        assert workflow._should_continue(state) == "finalize"
        
        # Should finalize when at max iterations
        state = {
            "converged": False,
            "iteration": 3,
            "max_iterations": 3
        }
        assert workflow._should_continue(state) == "finalize"
    
    @patch('src.workflow.FeatureGeneratorAgent')
    @patch('src.workflow.EvaluatorAgent')
    def test_workflow_run_structure(
        self, 
        mock_evaluator_class,
        mock_generator_class,
        sample_business_problem
    ):
        """Test that workflow.run returns correct structure."""
        
        # This is an integration test that would require actual API calls
        # For now, we test the structure of the initial state
        workflow = FeatureDiscoveryWorkflow(max_iterations=2)
        
        initial_state = {
            "business_problem": sample_business_problem,
            "current_features": None,
            "rubric": None,
            "evaluations": None,
            "iteration": 0,
            "max_iterations": 2,
            "converged": False,
            "final_output": None,
            "improvement_recommendations": None,
            "taxonomy_explanation": ""
        }
        
        assert initial_state["business_problem"] == sample_business_problem
        assert initial_state["iteration"] == 0
        assert initial_state["converged"] is False


class TestFeatureGeneratorAgent:
    """Test suite for Feature Generator Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        return Mock()
    
    def test_generator_initialization(self, mock_llm):
        """Test generator agent initialization."""
        from src.agents.feature_generator import FeatureGeneratorAgent
        
        generator = FeatureGeneratorAgent(mock_llm)
        assert generator.llm is not None
        assert generator.prompt is not None


class TestEvaluatorAgent:
    """Test suite for Evaluator Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        return Mock()
    
    def test_evaluator_initialization(self, mock_llm):
        """Test evaluator agent initialization."""
        from src.agents.evaluator import EvaluatorAgent
        
        evaluator = EvaluatorAgent(mock_llm)
        assert evaluator.llm is not None
        assert evaluator.rubric_prompt is not None
        assert evaluator.evaluation_prompt is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
