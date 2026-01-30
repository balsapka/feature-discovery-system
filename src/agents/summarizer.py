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

        system_message = """You are an expert at extracting key information from business problem descriptions.

Your task is to condense a long business problem description into a concise summary that preserves:
1. The core prediction/classification objective
2. Key business context and constraints
3. Target variable and timeframe
4. Important domain-specific details

Keep the summary under 300 words. Focus on information relevant for feature engineering.
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

        if self.should_summarize(business_problem):
            summarized = self.summarize(business_problem)
            return {
                "business_problem_original": business_problem,
                "business_problem_summarized": summarized,
            }
        else:
            # No summarization needed, use original
            return {
                "business_problem_original": business_problem,
                "business_problem_summarized": business_problem,
            }
