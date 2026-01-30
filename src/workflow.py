"""
LangGraph workflow orchestration for the Feature Discovery System.
"""
from typing import Dict, Any, Optional, Literal, TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import operator
import os
from dotenv import load_dotenv

from .agents import FeatureGeneratorAgent, EvaluatorAgent, SummarizerAgent
from .models.schemas import WorkflowState, Feature, EvaluationRubric, FeatureScore


class GraphState(TypedDict):
    """State schema for the workflow graph."""
    business_problem: str
    business_problem_original: Optional[str]
    business_problem_summarized: Optional[str]
    current_features: Optional[List[Feature]]
    kept_features: Optional[List[Feature]]  # High-scoring features to keep
    rubric: Optional[EvaluationRubric]
    evaluations: Optional[List[FeatureScore]]
    iteration: int
    max_iterations: int
    converged: bool
    final_output: Optional[Dict[str, Any]]
    improvement_recommendations: Optional[str]
    low_scoring_count: Optional[int]  # Number of features to regenerate
    low_scoring_type_counts: Optional[Dict[str, int]]  # Data type distribution of low-scoring features
    taxonomy_explanation: str


class FeatureDiscoveryWorkflow:
    """
    Orchestrates the feature discovery workflow using LangGraph.
    Implements an evaluator-optimizer pattern with iterative refinement.
    """
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        model_name: Optional[str] = None,
        max_iterations: int = 3,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        compact_mode: bool = False,
        parallel: bool = False,
        score_threshold: float = 7.0
    ):
        """
        Initialize the workflow.

        Args:
            llm_provider: 'openai' or 'anthropic'
            model_name: Specific model name (e.g., 'claude-sonnet-4-5-20250929', 'gpt-4')
            max_iterations: Maximum number of generator-evaluator iterations
            temperature: LLM temperature for generation
            api_key: Optional API key (otherwise loads from .env)
            compact_mode: If True, use minimal prompts/schemas for faster execution
            parallel: If True, generate features in parallel batches (~2x speedup)
            score_threshold: Minimum average score to stop iterating (default 7.0)
        """
        load_dotenv()

        self.llm_provider = llm_provider
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.compact_mode = compact_mode
        self.parallel = parallel
        self.score_threshold = score_threshold

        # Initialize LLM
        self.llm = self._initialize_llm(llm_provider, model_name, api_key)

        # Initialize agents
        self.summarizer = SummarizerAgent(self.llm)
        self.generator = FeatureGeneratorAgent(self.llm, compact_mode=compact_mode, parallel=parallel)
        self.evaluator = EvaluatorAgent(self.llm, compact_mode=compact_mode, score_threshold=score_threshold)

        # Build the graph
        self.graph = self._build_graph()
    
    def _initialize_llm(
        self, 
        provider: str, 
        model_name: Optional[str],
        api_key: Optional[str]
    ):
        """Initialize the language model."""
        
        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            model = model_name or os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
            return ChatOpenAI(
                model=model,
                temperature=self.temperature,
                api_key=api_key
            )
        elif provider == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            model = model_name or os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929")
            return ChatAnthropic(
                model=model,
                temperature=self.temperature,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Create the graph with our state schema
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("finalize", self._finalize_node)

        # Add edges - start with summarization
        workflow.set_entry_point("summarize")
        workflow.add_edge("summarize", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edge: either continue iterating or finalize
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "continue": "generate",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _summarize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for summarizing long business problems."""
        return self.summarizer(state)

    def _generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for feature generation."""
        iteration = state.get("iteration", 0)
        kept_features = state.get("kept_features", [])
        low_scoring_count = state.get("low_scoring_count", 0)

        print(f"\n{'#'*60}")
        print(f"# GENERATION PHASE - Iteration {iteration + 1}/{state.get('max_iterations', self.max_iterations)}")
        print(f"{'#'*60}")

        # Show summarized business problem on first iteration
        if iteration == 0 and state.get("business_problem_summarized"):
            print(f"Business Problem (summarized): {state['business_problem_summarized']}")

        if iteration > 0:
            if kept_features and low_scoring_count:
                print(f"Mode: REGENERATION - keeping {len(kept_features)} features, generating {low_scoring_count} replacements")
            print(f"Feedback: {state.get('improvement_recommendations', 'None')}")

        # Use summarized version for generation
        gen_state = state.copy()
        if state.get("business_problem_summarized"):
            gen_state["business_problem"] = state["business_problem_summarized"]

        result = self.generator(gen_state)
        print(f"Total features after generation: {len(result.get('current_features', []))}")
        return result

    def _evaluate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for feature evaluation."""
        iteration = state.get("iteration", 0)
        print(f"\n{'#'*60}")
        print(f"# EVALUATION PHASE - Iteration {iteration + 1}/{state.get('max_iterations', self.max_iterations)}")
        print(f"{'#'*60}")

        # Use summarized version for evaluation
        eval_state = state.copy()
        if state.get("business_problem_summarized"):
            eval_state["business_problem"] = state["business_problem_summarized"]
        return self.evaluator(eval_state)
    
    def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for finalizing results."""

        print(f"\n{'#'*60}")
        print(f"# FINALIZATION PHASE")
        print(f"{'#'*60}")
        print(f"Total iterations completed: {state['iteration']}")
        print(f"Final feature count: {len(state['current_features'])}")

        # Calculate final average score
        if state.get("evaluations"):
            avg_score = sum(e.overall_score for e in state["evaluations"]) / len(state["evaluations"])
            print(f"Final average score: {avg_score:.2f}")

        # Package final output
        final_output = {
            "features": [f.model_dump() for f in state["current_features"]],
            "rubric": state["rubric"].model_dump(),
            "evaluations": [e.model_dump() for e in state["evaluations"]],
            "taxonomy_explanation": state.get("taxonomy_explanation", ""),
            "total_iterations": state["iteration"]
        }

        return {"final_output": final_output, "converged": True}
    
    def _should_continue(
        self,
        state: Dict[str, Any]
    ) -> Literal["continue", "finalize"]:
        """
        Decide whether to continue iterating or finalize.

        Args:
            state: Current workflow state

        Returns:
            "continue" if should iterate, "finalize" if done
        """
        if state["converged"] or state["iteration"] >= self.max_iterations:
            print(f"\n>>> WORKFLOW DECISION: FINALIZE (converged={state['converged']}, iteration={state['iteration']}/{self.max_iterations})")
            return "finalize"
        print(f"\n>>> WORKFLOW DECISION: CONTINUE to iteration {state['iteration'] + 1}")
        return "continue"
    
    def run(self, business_problem: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the feature discovery workflow.
        
        Args:
            business_problem: Natural language description of the business problem
            verbose: If True, print intermediate states
            
        Returns:
            Dictionary containing:
                - features: List of generated features
                - rubric: Evaluation rubric
                - evaluations: Feature scores
                - taxonomy_explanation: How features are organized
                - total_iterations: Number of iterations performed
        """
        
        # Initialize state
        initial_state = {
            "business_problem": business_problem,
            "business_problem_original": None,
            "business_problem_summarized": None,
            "current_features": None,
            "kept_features": None,
            "rubric": None,
            "evaluations": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "converged": False,
            "final_output": None,
            "improvement_recommendations": None,
            "low_scoring_count": None,
            "low_scoring_type_counts": None,
            "taxonomy_explanation": ""
        }
        
        # Run the graph
        if verbose:
            print(f"Starting Feature Discovery Workflow")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Business Problem: {business_problem}\n")
        
        final_state = self.graph.invoke(initial_state)
        
        if verbose:
            print(f"\nWorkflow completed after {final_state['iteration']} iterations")
        
        return final_state["final_output"]
    
    def visualize(self, output_path: str = "workflow_graph.png"):
        """
        Visualize the workflow graph.
        
        Args:
            output_path: Path to save the visualization
        """
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except ImportError:
            print("To visualize the graph, install: pip install pygraphviz")
