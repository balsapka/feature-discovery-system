"""
Feature Generator Agent - Generates candidate features from business problems.
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..models.schemas import GeneratorOutput, Feature, DataType


class FeatureGeneratorAgent:
    """
    Agent responsible for generating candidate features based on a business problem.
    """
    
    def __init__(self, llm):
        """
        Initialize the Feature Generator Agent.
        
        Args:
            llm: Language model instance (OpenAI or Anthropic)
        """
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=GeneratorOutput)
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for feature generation."""
        
        system_message = """You are an expert data scientist working in a banking institution. 
Your role is to generate relevant candidate features for predictive modeling based on business problems.

Given a business problem, you should:
1. Identify key data sources and feature categories
2. Generate raw feature concepts (not aggregated/transformed features yet)
3. Consider different data types: structured (transactional, demographic), unstructured (documents, text), 
   time-series (historical patterns), and external (macroeconomic, market data)
4. Create a clear taxonomy to organize the features
5. Provide rationale for why each feature is relevant

Focus on feature concepts that would be found in a banking Feature Store.

IMPORTANT: Your response must be valid JSON matching this exact schema:
{{
    "features": [
        {{
            "name": "feature_name",
            "description": "detailed description",
            "data_type": "structured|unstructured|time_series|external",
            "taxonomy": "category/subcategory",
            "rationale": "why this feature is relevant"
        }}
    ],
    "taxonomy_explanation": "explanation of how features are organized"
}}

Generate 8-15 diverse, high-quality feature concepts."""

        human_message = """Business Problem:
{business_problem}

{feedback}

Generate candidate features as a JSON object following the specified schema."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def generate(
        self, 
        business_problem: str, 
        feedback: str = ""
    ) -> GeneratorOutput:
        """
        Generate candidate features for a business problem.
        
        Args:
            business_problem: Natural language description of the business problem
            feedback: Optional feedback from previous evaluation for refinement
            
        Returns:
            GeneratorOutput containing features and taxonomy explanation
        """
        
        # Add feedback context if provided
        feedback_text = ""
        if feedback:
            feedback_text = f"""
Previous Iteration Feedback:
{feedback}

Please incorporate this feedback to improve the feature list.
"""
        
        # Create the chain
        chain = self.prompt | self.llm | self.parser
        
        # Generate features
        result = chain.invoke({
            "business_problem": business_problem,
            "feedback": feedback_text
        })
        
        # Parse into Pydantic model
        return GeneratorOutput(**result)
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated features
        """
        business_problem = state["business_problem"]
        
        # Get feedback from previous iteration if available
        feedback = ""
        if state.get("evaluations") and state["iteration"] > 0:
            # Extract improvement recommendations from last evaluation
            if state.get("improvement_recommendations"):
                feedback = state["improvement_recommendations"]
        
        # Generate features
        output = self.generate(business_problem, feedback)
        
        # Update state
        return {
            "current_features": output.features,
            "taxonomy_explanation": output.taxonomy_explanation
        }
