"""
Feature Generator Agent - Generates candidate features from business problems.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..models.schemas import GeneratorOutput, CompactGeneratorOutput, Feature, CompactFeature, DataType


class FeatureGeneratorAgent:
    """
    Agent responsible for generating candidate features based on a business problem.
    Supports parallel generation for faster execution.
    """

    def __init__(self, llm, compact_mode: bool = False, parallel: bool = False, num_batches: int = 2):
        """
        Initialize the Feature Generator Agent.

        Args:
            llm: Language model instance (OpenAI or Anthropic)
            compact_mode: If True, use minimal schema for faster generation
            parallel: If True, generate features in parallel batches
            num_batches: Number of parallel batches (default 2)
        """
        self.llm = llm
        self.compact_mode = compact_mode
        self.parallel = parallel
        self.num_batches = num_batches

        if compact_mode:
            self.parser = JsonOutputParser(pydantic_object=CompactGeneratorOutput)
            self.prompt = self._create_compact_prompt()
            self.parallel_prompt = self._create_compact_parallel_prompt()
        else:
            self.parser = JsonOutputParser(pydantic_object=GeneratorOutput)
            self.prompt = self._create_prompt()
            self.parallel_prompt = self._create_parallel_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for feature generation (full mode)."""

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

    def _create_parallel_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for parallel batch generation (full mode)."""

        system_message = """You are an expert data scientist working in a banking institution.
Your role is to generate relevant candidate features for predictive modeling.

Focus on {focus_area} features. Generate ONLY features in this category.

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

Generate 8-10 diverse, high-quality feature concepts in your assigned category."""

        human_message = """Business Problem:
{business_problem}

{feedback}

Generate {focus_area} features as JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _create_compact_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for feature generation (compact mode)."""

        system_message = """You are a data scientist. Generate feature names for predictive modeling.

Output JSON with this EXACT format:
{{
    "features": [
        {{"name": "feature_name", "desc": "brief description (max 15 words)", "type": "structured|time_series|unstructured|external"}}
    ]
}}

Types: structured (demographic, account), time_series (historical), unstructured (text), external (market data)

Generate 15-20 diverse features. Keep descriptions SHORT."""

        human_message = """Problem: {business_problem}

{feedback}

Generate features as JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _create_compact_parallel_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for parallel batch generation (compact mode)."""

        system_message = """Generate {focus_area} features ONLY. Output JSON:
{{
    "features": [
        {{"name": "feature_name", "desc": "brief (max 15 words)", "type": "structured|time_series|unstructured|external"}}
    ]
}}

Generate 10 features in your category. Keep descriptions SHORT."""

        human_message = """Problem: {business_problem}
{feedback}
Generate {focus_area} features as JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _get_focus_areas(self) -> List[str]:
        """Get focus areas for parallel generation."""
        return [
            "behavioral and transactional",
            "demographic and external"
        ]

    def _generate_batch(
        self,
        business_problem: str,
        feedback_text: str,
        focus_area: str
    ) -> List[Feature]:
        """Generate a single batch of features for a focus area."""
        chain = self.parallel_prompt | self.llm | self.parser

        result = chain.invoke({
            "business_problem": business_problem,
            "feedback": feedback_text,
            "focus_area": focus_area
        })

        if self.compact_mode:
            compact_output = CompactGeneratorOutput(**result)
            return [f.to_feature() for f in compact_output.features]
        else:
            output = GeneratorOutput(**result)
            return output.features

    def generate_parallel(
        self,
        business_problem: str,
        feedback: str = ""
    ) -> GeneratorOutput:
        """
        Generate features in parallel batches.

        Args:
            business_problem: Natural language description of the business problem
            feedback: Optional feedback from previous evaluation

        Returns:
            GeneratorOutput containing combined features
        """
        feedback_text = ""
        if feedback:
            feedback_text = f"Feedback: {feedback}" if self.compact_mode else f"Previous feedback: {feedback}"

        focus_areas = self._get_focus_areas()

        # Run batches in parallel using ThreadPoolExecutor
        all_features = []
        with ThreadPoolExecutor(max_workers=self.num_batches) as executor:
            futures = [
                executor.submit(
                    self._generate_batch,
                    business_problem,
                    feedback_text,
                    focus_area
                )
                for focus_area in focus_areas
            ]

            for future in futures:
                try:
                    batch_features = future.result()
                    all_features.extend(batch_features)
                except Exception as e:
                    print(f"Batch generation failed: {e}")

        # Deduplicate by feature name
        seen_names = set()
        unique_features = []
        for f in all_features:
            if f.name not in seen_names:
                seen_names.add(f.name)
                unique_features.append(f)

        return GeneratorOutput(
            features=unique_features,
            taxonomy_explanation="Features organized by behavioral/transactional and demographic/external categories"
        )

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
        if self.parallel:
            return self.generate_parallel(business_problem, feedback)

        # Add feedback context if provided
        feedback_text = ""
        if feedback:
            feedback_text = f"Feedback: {feedback}" if self.compact_mode else f"""
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

        if self.compact_mode:
            # Convert compact output to full output
            compact_output = CompactGeneratorOutput(**result)
            features = [f.to_feature() for f in compact_output.features]
            return GeneratorOutput(features=features, taxonomy_explanation="")
        else:
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
