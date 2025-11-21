"""
Advanced example showing the agent's self-evolution capabilities.

This demonstrates:
- Autonomous package installation
- Dynamic tool creation
- Complex research workflows
"""

import os
from msr_scientist import ResearchAgent


def main():
    # Create agent
    agent = ResearchAgent(verbose=True)

    # Complex research task that will require the agent to evolve
    task = """
    Research Task: Analyze the effectiveness of different text similarity metrics.

    Context:
    You need to compare cosine similarity, Jaccard similarity, and Levenshtein distance
    for measuring text similarity across different types of text pairs.

    Important: You will need to install any required packages (like python-Levenshtein,
    scikit-learn, etc.) autonomously. If you need specialized tools, create them.

    Requirements:
    1. Formulate hypotheses about which metrics work best for which text types
    2. Create test datasets (short texts, long documents, code snippets)
    3. Implement or install the necessary similarity metrics
    4. Run comparative experiments
    5. Generate visualizations (create a plotting tool if needed)
    6. Draft a research paper with:
       - Abstract
       - Introduction
       - Methodology
       - Results with tables
       - Conclusion

    Follow your evolution cycle and create any tools you need along the way.
    """

    # Let the agent evolve and conduct research
    result = agent.research(task)

    print("\n✅ Advanced research complete!")
    print("The agent should have:")
    print("  - Installed required packages automatically")
    print("  - Created custom tools as needed")
    print("  - Generated a complete research paper")


if __name__ == "__main__":
    main()
