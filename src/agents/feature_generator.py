"""
Feature Generator Agent - Generates candidate features from business problems.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..models.schemas import GeneratorOutput, CompactGeneratorOutput, Feature


class FeatureGeneratorAgent:  # pylint: disable=too-many-instance-attributes
    """
    Agent responsible for generating candidate features based on a business problem.
    Supports parallel generation for faster execution.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        llm,
        compact_mode: bool = False,
        parallel: bool = False,
        num_batches: int = 2,
        target_feature_count: int = 20
    ):
        """
        Initialize the Feature Generator Agent.

        Args:
            llm: Language model instance (OpenAI or Anthropic)
            compact_mode: If True, use minimal schema for faster generation
            parallel: If True, generate features in parallel batches
            num_batches: Number of parallel batches (default 2)
            target_feature_count: Target number of features to generate (default 20)
        """
        self.llm = llm
        self.compact_mode = compact_mode
        self.parallel = parallel
        self.num_batches = num_batches
        self.target_feature_count = target_feature_count

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
2. Generate feature concepts (both raw attributes and aggregated/derived features as appropriate)
3. Consider different data types: structured (transactional, demographic), unstructured (documents, text),
   time-series (historical patterns), and external (macroeconomic, market data)
4. Create a clear taxonomy to organize the features
5. Provide rationale for why each feature is relevant

**CRITICAL: If the business problem contains a "USER-REQUESTED FEATURES" section or explicitly mentions
specific features/data sources the user wants, you MUST include those features FIRST in your output.
These are non-negotiable requirements from the user.**

Focus on feature concepts that would be found in a banking Feature Store.

IMPORTANT: Feature names MUST be lowercase with underscores (snake_case), e.g., "monthly_income", "avg_transaction_amount".

IMPORTANT: Your response must be valid JSON matching this exact schema:
{{
    "features": [
        {{
            "name": "feature_name_in_lowercase",
            "description": "detailed description",
            "data_type": "structured|unstructured|time_series|external",
            "taxonomy": "category/subcategory",
            "rationale": "why this feature is relevant"
        }}
    ],
    "taxonomy_explanation": "explanation of how features are organized"
}}

{num_features_instruction}"""

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
        features_per_batch = (self.target_feature_count + self.num_batches - 1) // self.num_batches

        system_message = f"""You are an expert data scientist working in a banking institution.
Your role is to generate relevant candidate features for predictive modeling.

Focus on {{focus_area}} features. Generate ONLY features in this category.

IMPORTANT: Feature names MUST be lowercase with underscores (snake_case), e.g., "monthly_income", "avg_transaction_amount".

IMPORTANT: Your response must be valid JSON matching this exact schema:
{{{{
    "features": [
        {{{{
            "name": "feature_name_in_lowercase",
            "description": "detailed description",
            "data_type": "structured|unstructured|time_series|external",
            "taxonomy": "category/subcategory",
            "rationale": "why this feature is relevant"
        }}}}
    ],
    "taxonomy_explanation": "explanation of how features are organized"
}}}}

CRITICAL: Generate EXACTLY {features_per_batch} diverse, high-quality features \
in your assigned category."""

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

        system_message = """You are a data scientist. Generate feature names for predictive \
modeling.

**CRITICAL: If the problem mentions "USER-REQUESTED FEATURES" or specific features the user wants,
include those FIRST. These are mandatory requirements.**

Feature names MUST be lowercase with underscores (snake_case), e.g., "monthly_income", "avg_transaction_amount".

Output JSON with this EXACT format:
{{
    "features": [
        {{"name": "feature_name_lowercase", "desc": "brief description (max 15 words)", "type": "structured|time_series|unstructured|external"}}
    ]
}}

Types: structured (demographic, account), time_series (historical), unstructured (text), external (market data)

{num_features_instruction}"""

        human_message = """Problem: {business_problem}

{feedback}

Generate features as JSON."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def _create_compact_parallel_prompt(self) -> ChatPromptTemplate:
        """Create the prompt for parallel batch generation (compact mode)."""
        features_per_batch = (self.target_feature_count + self.num_batches - 1) // self.num_batches

        system_message = f"""Generate {{focus_area}} features ONLY. Feature names MUST be \
lowercase with underscores (snake_case).

Output JSON:
{{{{
    "features": [
        {{{{"name": "feature_name_lowercase", "desc": "brief (max 15 words)", "type": "structured|time_series|unstructured|external"}}}}
    ]
}}}}

CRITICAL: Generate EXACTLY {features_per_batch} features in your category. \
Keep descriptions SHORT."""

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

    def _normalize_feature_name(self, name: str) -> str:
        """Normalize feature name to lowercase snake_case."""
        return name.lower().replace(" ", "_").replace("-", "_")

    def _generate_batch(
        self,
        business_problem: str,
        feedback_text: str,
        focus_area: str
    ) -> List[Feature]:
        """Generate a single batch of features for a focus area."""
        # pylint: disable=unsupported-binary-operation
        chain = self.parallel_prompt | self.llm | self.parser

        result = chain.invoke({
            "business_problem": business_problem,
            "feedback": feedback_text,
            "focus_area": focus_area
        })

        if self.compact_mode:
            compact_output = CompactGeneratorOutput(**result)
            # Normalization happens in CompactFeature.to_feature()
            return [f.to_feature() for f in compact_output.features]

        output = GeneratorOutput(**result)
        # Normalize feature names in full mode
        for feature in output.features:
            feature.name = self._normalize_feature_name(feature.name)
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
            if self.compact_mode:
                feedback_text = f"Feedback: {feedback}"
            else:
                feedback_text = f"Previous feedback: {feedback}"

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
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"Batch generation failed: {exc}")

        # Deduplicate by feature name
        seen_names = set()
        unique_features = []
        for f in all_features:
            if f.name not in seen_names:
                seen_names.add(f.name)
                unique_features.append(f)

        return GeneratorOutput(
            features=unique_features,
            taxonomy_explanation=(
                "Features organized by behavioral/transactional and "
                "demographic/external categories"
            )
        )

    def generate(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        business_problem: str,
        feedback: str = "",
        num_features: int = None,
        type_distribution: dict = None,
        excluded_names: set = None
    ) -> GeneratorOutput:
        """
        Generate candidate features for a business problem.

        Args:
            business_problem: Natural language description of the business problem
            feedback: Optional feedback from previous evaluation for refinement
            num_features: Optional specific number of features to generate (for regeneration)
            type_distribution: Optional dict of data_type -> count to maintain variety
            excluded_names: Optional set of feature names to avoid (already tried or kept)

        Returns:
            GeneratorOutput containing features and taxonomy explanation
        """
        if self.parallel and num_features is None:
            # Only use parallel generation for initial full generation
            return self.generate_parallel(business_problem, feedback)

        # Add feedback context if provided
        feedback_text = ""
        if feedback:
            feedback_text = f"Feedback: {feedback}" if self.compact_mode else f"""
Previous Iteration Feedback:
{feedback}

Please incorporate this feedback to improve the feature list.
"""

        # Add exclusion list to prevent regenerating same features
        if excluded_names:
            excluded_list = ", ".join(sorted(excluded_names))
            exclusion_msg = (
                f"\n\nDO NOT generate any of these features "
                f"(already exist or were rejected): {excluded_list}"
            )
            feedback_text += exclusion_msg

        # Build the num_features instruction for the system prompt
        if num_features is not None:
            if type_distribution:
                type_instructions = ", ".join(
                    [f"{count} {dtype}" for dtype, count in type_distribution.items()]
                )
                num_features_instruction = (
                    f"CRITICAL: Generate EXACTLY {num_features} features "
                    f"(no more, no less). Maintain data type variety: {type_instructions}."
                )
            else:
                num_features_instruction = (
                    f"CRITICAL: Generate EXACTLY {num_features} features (no more, no less)."
                )
        else:
            num_features_instruction = (
                f"CRITICAL: Generate EXACTLY {self.target_feature_count} diverse, "
                "high-quality features (no more, no less)."
            )
            if self.compact_mode:
                num_features_instruction += " Keep descriptions SHORT."

        # Create the chain
        chain = self.prompt | self.llm | self.parser  # pylint: disable=unsupported-binary-operation

        # Generate features
        result = chain.invoke({
            "business_problem": business_problem,
            "feedback": feedback_text,
            "num_features_instruction": num_features_instruction
        })

        if self.compact_mode:
            # Convert compact output to full output
            # Normalization happens in CompactFeature.to_feature()
            compact_output = CompactGeneratorOutput(**result)
            features = [f.to_feature() for f in compact_output.features]
            return GeneratorOutput(features=features, taxonomy_explanation="")

        output = GeneratorOutput(**result)
        # Normalize feature names in full mode
        for feature in output.features:
            feature.name = self._normalize_feature_name(feature.name)
        return output

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function.

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated features
        """
        business_problem = state["business_problem"]
        iteration = state.get("iteration", 0)

        # Get feedback from previous iteration if available
        feedback = ""
        if state.get("evaluations") and iteration > 0:
            # Extract improvement recommendations from last evaluation
            if state.get("improvement_recommendations"):
                feedback = state["improvement_recommendations"]

        # Check if we need to regenerate only a subset of features
        kept_features = state.get("kept_features", [])
        low_scoring_count = state.get("low_scoring_count")
        low_scoring_type_counts = state.get("low_scoring_type_counts")

        if kept_features and low_scoring_count and low_scoring_count > 0:
            # Regeneration mode: only generate replacements for low-scoring features
            kept_feature_names = {f.name for f in kept_features}

            # Get rejected feature names to prevent regenerating them
            rejected_names = state.get("rejected_feature_names", set())

            # Combine kept + rejected as exclusion list
            excluded_names = kept_feature_names | rejected_names

            print(f"Regeneration mode: keeping {len(kept_features)} features, "
                  f"generating {low_scoring_count} new ones")
            print(f"Excluding {len(excluded_names)} names (kept + previously rejected)")
            if low_scoring_type_counts:
                print(f"Type distribution to match: {low_scoring_type_counts}")

            # Generate only the needed number of replacement features
            output = self.generate(
                business_problem,
                feedback,
                num_features=low_scoring_count,
                type_distribution=low_scoring_type_counts,
                excluded_names=excluded_names
            )

            # Filter out any duplicates that slipped through (LLM might ignore instructions)
            new_features = [f for f in output.features if f.name not in excluded_names]
            if len(new_features) < len(output.features):
                print(f"Filtered {len(output.features) - len(new_features)} duplicate features")

            # Combine kept features with new features
            combined_features = kept_features + new_features
            print(f"Combined: {len(kept_features)} kept + {len(new_features)} new = "
                  f"{len(combined_features)} total")

            return {
                "current_features": combined_features,
                "taxonomy_explanation": output.taxonomy_explanation,
                # Pass kept_features AND their names for reliable matching in evaluator
                "kept_feature_names": kept_feature_names,
                "low_scoring_count": None,
                "low_scoring_type_counts": None
            }

        # Initial generation: generate full set of features
        output = self.generate(business_problem, feedback)

        return {
            "current_features": output.features,
            "taxonomy_explanation": output.taxonomy_explanation
        }
