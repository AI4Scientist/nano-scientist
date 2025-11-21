"""
Example showing how to add custom tools to the research agent.
"""

from smolagents import tool
from msr_scientist import ResearchAgent


# Define a custom tool for your specific research domain
@tool
def fetch_arxiv_papers(query: str, max_results: int = 5) -> str:
    """
    Fetch recent papers from arXiv based on a search query.

    Args:
        query: Search query string
        max_results: Maximum number of papers to return

    Returns:
        JSON string with paper metadata
    """
    import json
    # This is a placeholder - in reality would call arXiv API
    papers = {
        "query": query,
        "results": [
            {
                "title": f"Paper about {query}",
                "authors": ["Researcher A", "Researcher B"],
                "abstract": "This paper investigates...",
                "url": "https://arxiv.org/abs/2401.00000"
            }
        ]
    }
    return json.dumps(papers, indent=2)


@tool
def calculate_statistics(data: str) -> str:
    """
    Calculate basic statistics from a JSON array of numbers.

    Args:
        data: JSON array of numbers

    Returns:
        JSON string with mean, median, std dev, etc.
    """
    import json
    import statistics

    numbers = json.loads(data)
    stats = {
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
        "min": min(numbers),
        "max": max(numbers)
    }
    return json.dumps(stats, indent=2)


def main():
    # Create agent with custom tools
    agent = ResearchAgent(
        additional_tools=[fetch_arxiv_papers, calculate_statistics],
        verbose=True
    )

    task = """
    Research the recent trends in machine learning interpretability.

    1. Use fetch_arxiv_papers to find relevant papers
    2. Analyze the trends you find
    3. If you need additional tools for analysis, create them
    4. Draft a survey paper summarizing the findings
    """

    result = agent.research(task)
    print("Research with custom tools complete!")


if __name__ == "__main__":
    main()
