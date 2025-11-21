"""
Basic example of using the MSR-Scientist research agent.

This demonstrates a simple research workflow where the agent:
1. Generates hypotheses
2. Runs experiments
3. Drafts a research paper
"""

import os
from msr_scientist import ResearchAgent

# Make sure to set your HuggingFace token
# export HF_TOKEN=your_token_here
# Or set it programmatically (not recommended for production)
# os.environ["HF_TOKEN"] = "your_token_here"


def main():
    # Create the research agent
    agent = ResearchAgent(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        verbose=True
    )

    # Define a research task
    task = """
    Research Task: Investigate the relationship between sorting algorithm complexity and input size.

    Context:
    - We want to empirically verify Big-O complexity claims
    - Focus on bubble sort, merge sort, and quicksort
    - Test with arrays of sizes: 100, 500, 1000, 5000

    Objectives:
    1. Generate hypotheses about performance differences
    2. Implement timing experiments for each algorithm
    3. Collect and analyze results
    4. Draft a brief research paper with findings

    Please follow your self-evolution cycle: plan → implement → verify → ship
    """

    # Run the research
    result = agent.research(task)

    print("\n" + "="*60)
    print("Research completed!")
    print("Check the outputs/ directory for:")
    print("  - outputs/experiments/ for experiment results")
    print("  - outputs/papers/ for generated papers")
    print("  - outputs/evolution/ for evolution state")
    print("="*60)


if __name__ == "__main__":
    main()
