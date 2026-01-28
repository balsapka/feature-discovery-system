"""
LangGraph workflow orchestration for the Feature Discovery System.
"""
from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

from .agents import FeatureGeneratorAgent, EvaluatorAgent
from .models.schemas import WorkflowState


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
        api_key: Optional[str] = None
    ):
        """
        Initialize the workflow.
        
        Args:
            llm_provider: 'openai' or 'anthropic'
            model_name: Specific model name (e.g., 'claude-sonnet-4-5-20250929', 'gpt-4')
            max_iterations: Maximum number of generator-evaluator iterations
            temperature: LLM temperature for generation
            api_key: Optional API key (otherwise loads from .env)
        """
        load_dotenv()
        
        self.llm_provider = llm_provider
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = self._initialize_llm(llm_provider, model_name, api_key)
        
        # Initialize agents
        self.generator = FeatureGeneratorAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm)
        
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
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("generate")
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
    
    def _generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for feature generation."""
        return self.generator(state)
    
    def _evaluate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for feature evaluation."""
        return self.evaluator(state)
    
    def _finalize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node for finalizing results."""
        
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
            return "finalize"
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
            "current_features": None,
            "rubric": None,
            "evaluations": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "converged": False,
            "final_output": None,
            "improvement_recommendations": None,
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
