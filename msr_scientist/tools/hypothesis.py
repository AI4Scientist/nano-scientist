"""
Hypothesis generation tool for research tasks.
"""

from smolagents import tool
from typing import Dict, List
import json


@tool
def generate_hypotheses(
    task_description: str,
    research_context: str = "",
    num_hypotheses: int = 3
) -> str:
    """
    Generate research hypotheses based on a task description and context.

    This tool analyzes the research task and context to formulate testable
    hypotheses or research directions. It returns structured hypotheses
    that can guide experimentation.

    Args:
        task_description: The main research question or task to investigate
        research_context: Additional context, background, or constraints
        num_hypotheses: Number of hypotheses to generate (default: 3)

    Returns:
        A JSON string containing the generated hypotheses with their rationales
    """
    # This is a placeholder that the agent will use to structure hypothesis generation
    # In practice, the CodeAgent will use this tool by calling it with appropriate arguments
    # and the LLM will generate the actual hypotheses based on the task

    hypotheses = {
        "task": task_description,
        "context": research_context,
        "hypotheses": [],
        "metadata": {
            "requested_count": num_hypotheses,
            "status": "generated"
        }
    }

    # The actual hypothesis generation will be done by the LLM through the agent
    # This structure provides a template for organizing the output
    return json.dumps(hypotheses, indent=2)


@tool
def refine_hypothesis(
    hypothesis: str,
    feedback: str,
    experimental_results: str = ""
) -> str:
    """
    Refine a hypothesis based on feedback or experimental results.

    Args:
        hypothesis: The original hypothesis to refine
        feedback: Feedback or observations about the hypothesis
        experimental_results: Results from experiments testing the hypothesis

    Returns:
        A refined version of the hypothesis
    """
    refinement = {
        "original_hypothesis": hypothesis,
        "feedback": feedback,
        "experimental_results": experimental_results,
        "refined_hypothesis": "",
        "changes_made": []
    }

    return json.dumps(refinement, indent=2)


class HypothesisGenerator:
    """
    A collection of hypothesis generation utilities.

    This class provides a simple interface for hypothesis-related operations
    in the research workflow.
    """

    def __init__(self):
        self.hypotheses_history = []

    def generate(self, task: str, context: str = "", count: int = 3) -> Dict:
        """Generate hypotheses for a research task."""
        result = generate_hypotheses(task, context, count)
        hypothesis_dict = json.loads(result)
        self.hypotheses_history.append(hypothesis_dict)
        return hypothesis_dict

    def refine(self, hypothesis: str, feedback: str, results: str = "") -> Dict:
        """Refine a hypothesis based on feedback."""
        result = refine_hypothesis(hypothesis, feedback, results)
        return json.loads(result)

    def get_history(self) -> List[Dict]:
        """Get the history of generated hypotheses."""
        return self.hypotheses_history
