"""
Summarizer Agent - Condenses long business problems into key points.
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate


class SummarizerAgent:
    """
    Agent responsible for summarizing long business problems.
    Extracts key information to reduce input tokens for downstream agents.
    """

    # Threshold in characters - summarize if input exceeds this
    CHAR_THRESHOLD = 500

    def __init__(self, llm):
        """
        Initialize the Summarizer Agent.

        Args:
            llm: Language model instance
        """
        self.llm = llm
        self.prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for summarization."""

        system_message = """You are an expert at extracting key information from business problem descriptions for ML feature engineering.

Your output has TWO parts:

PART 1 - SUMMARY (100-200 words max):
Concisely describe ONLY:
- The core prediction/classification objective
- Target variable and timeframe
- Key business context and constraints
- Important domain-specific details

DO NOT mention specific features or data sources in this part - those go in Part 2.

PART 2 - USER-REQUESTED FEATURES (if any):
If the user explicitly mentions specific features, data sources, or variables they want included, list them here.
Look for phrases like: "consider features such as...", "include...", "make sure to...", "features like...", "key indicators are...", "we should look at..."

Format:
USER-REQUESTED FEATURES: [comma-separated list of specific features/data sources]

If no specific features are mentioned, omit Part 2 entirely.

Output ONLY the summary, no preamble or explanation."""

        human_message = """Business Problem:
{business_problem}

Provide a concise summary:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def summarize(self, business_problem: str) -> str:
        """
        Summarize a long business problem.

        Args:
            business_problem: The full business problem description

        Returns:
            Condensed summary of the business problem
        """
        chain = self.prompt | self.llm
        result = chain.invoke({"business_problem": business_problem})

        # Extract content from AIMessage if needed
        if hasattr(result, 'content'):
            return result.content
        return str(result)

    def should_summarize(self, business_problem: str) -> bool:
        """
        Determine if the business problem needs summarization.

        Args:
            business_problem: The business problem text

        Returns:
            True if summarization is recommended
        """
        return len(business_problem) > self.CHAR_THRESHOLD

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function.

        Args:
            state: Current workflow state

        Returns:
            Updated state with summarized problem
        """
        business_problem = state["business_problem"]

        print(f"\n{'#'*60}")
        print(f"# SUMMARIZATION PHASE")
        print(f"{'#'*60}")
        print(f"Original problem length: {len(business_problem)} chars (threshold: {self.CHAR_THRESHOLD})")

        if self.should_summarize(business_problem):
            print("Summarizing business problem...")
            summarized = self.summarize(business_problem)
            print(f"Summarized to: {len(summarized)} chars")
            print(f"\nSummarized problem:\n{summarized}")
            return {
                "business_problem_original": business_problem,
                "business_problem_summarized": summarized,
            }
        else:
            # No summarization needed, use original
            print("No summarization needed, using original")
            return {
                "business_problem_original": business_problem,
                "business_problem_summarized": business_problem,
            }
