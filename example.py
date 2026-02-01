"""example.py - Usage examples for Mini Researcher Agent

Demonstrates different ways to use the research agent:
1. Basic programmatic API
2. Custom configuration
3. Error handling
4. CLI usage
"""

import os
import sys
from pathlib import Path
from src.main import research_and_plan, execute_and_analyze, report_and_write
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70 + "\n")

    from main import ResearchAgent

    # Initialize agent with defaults
    agent = ResearchAgent()

    # Run a simple research task
    task = (
        "Compare bubble sort and quicksort performance characteristics. "
        "Generate a research paper in PDF using the ACM template."
    )
    print(f"Task: {task}\n")

    result = agent(task)

    print(f"\nResult: {result}")


def example_2_custom_config():
    """Example 2: Custom configuration for more intensive research."""
    print("\n" + "="*70)
    print("Example 2: Custom Configuration")
    print("="*70 + "\n")

    from main import ResearchAgent

    # Set custom environment variables before initialization
    os.environ["STAGE1_MAX_LOOPS"] = "10"  # More research loops
    os.environ["STAGE2_STEP_LIMIT"] = "100"  # More agent steps
    os.environ["STAGE2_COST_LIMIT"] = "5.0"  # Higher cost limit

    agent = ResearchAgent(verbose=True)

    task = (
        "Research the latest advances in transformer attention mechanisms in 2024. "
        "Generate a research paper in PDF using the ACM template."
    )
    print(f"Task: {task}\n")

    result = agent(task)

    print(f"\nResult: {result}")


def example_3_minimal_config():
    """Example 3: Minimal configuration for quick experiments."""
    print("\n" + "="*70)
    print("Example 3: Minimal Configuration (Quick Mode)")
    print("="*70 + "\n")

    from main import ResearchAgent

    # Set minimal configuration
    os.environ["STAGE1_MAX_LOOPS"] = "2"  # Fewer research loops
    os.environ["STAGE2_STEP_LIMIT"] = "20"  # Fewer agent steps
    os.environ["STAGE2_COST_LIMIT"] = "0.5"  # Lower cost limit

    agent = ResearchAgent(verbose=True)

    task = (
        "Implement FizzBuzz with unit tests. "
        "Generate a research paper in PDF using the ACM template."
    )
    print(f"Task: {task}\n")

    result = agent(task)

    print(f"\nResult: {result}")


def example_4_with_error_handling():
    """Example 4: Error handling and validation."""
    print("\n" + "="*70)
    print("Example 4: Error Handling")
    print("="*70 + "\n")

    from main import ResearchAgent
    import json

    agent = ResearchAgent(verbose=False)

    task = (
        "Research Python list comprehensions performance. "
        "Generate a research paper in PDF using the ACM template."
    )
    print(f"Task: {task}\n")

    try:
        result = agent(task)
        print(f"Success! Result:\n{result}")

        # Parse result if it's JSON
        try:
            result_data = json.loads(result)
            if isinstance(result_data, dict):
                print(f"\nParsed result keys: {list(result_data.keys())}")
        except json.JSONDecodeError:
            print("\nResult is not JSON (plain text)")

    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("- Check that all API keys are set in .env")
        print("- Verify HF_TOKEN, OPENAI_API_KEY, TAVILY_API_KEY are valid")
        print("- Ensure mini-swe-agent and all dependencies are installed")


def example_5_programmatic_workflow():
    """Example 5: Step-by-step programmatic workflow."""
    print("\n" + "="*70)
    print("Example 5: Programmatic Workflow (Direct Tool Calls)")
    print("="*70 + "\n")

    task = (
        "Implement binary search and measure performance. "
        "Generate a research paper in PDF using the ACM template."
    )
    output_dir = "research_outputs"

    print("Step 1: Research and Planning...")
    stage1_result = research_and_plan(task, output_dir)
    stage1_data = json.loads(stage1_result)
    print(f"  ✓ Proposal file: {stage1_data['proposal_file']}")
    print(f"  ✓ Citations: {stage1_data['num_citations']}")
    print(f"  ✓ Hypotheses: {stage1_data['num_hypotheses']}")

    print("\nStep 2: Execute and Analyze...")
    stage2_result = execute_and_analyze(stage1_data['proposal_file'])
    stage2_data = json.loads(stage2_result)
    print(f"  ✓ Workspace: {stage2_data['workspace']}")
    print(f"  ✓ Cost: ${stage2_data['cost_usd']:.4f}")

    print("\nStep 3: Generate Report (ACM PDF)...")
    stage3_result = report_and_write(
        stage1_data['proposal_file'],
        stage2_data['workspace']
    )
    stage3_data = json.loads(stage3_result)
    print(f"  ✓ PDF: {stage3_data['pdf_path']}")

    print("\n" + "="*70)
    print(f"Complete! Final PDF: {stage3_data['pdf_path']}")
    print("="*70)


def example_6_matrix_subregion_sum():
    """Example 6: Algorithmic research - find a rectangular region in a matrix with a target sum."""
    print("\n" + "="*70)
    print("Example 6: Matrix Rectangular Region Sum")
    print("="*70 + "\n")

    from main import ResearchAgent

    agent = ResearchAgent(verbose=True)

    task = (
        "In a two-dimensional square matrix, find a rectangular region "
        "such that the sum of the numbers within the region equals a "
        "pre-given number. Research efficient algorithms for this problem, "
        "implement at least a brute-force and an optimized approach using "
        "prefix sums, compare their time complexity, and validate correctness "
        "with test cases. Generate a research paper in PDF using the ACM template."
    )
    print(f"Task: {task}\n")

    result = agent(task)

    print(f"\nResult: {result}")


def check_environment():
    """Check if environment is properly configured."""
    print("\n" + "="*70)
    print("Environment Check")
    print("="*70 + "\n")

    required_vars = [
        "HF_TOKEN",
        "OPENAI_API_KEY",
    ]

    optional_vars = [
        "TAVILY_API_KEY",
        "PERPLEXITY_API_KEY",
    ]

    all_good = True

    print("Required variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {'*' * 10} (set)")
        else:
            print(f"  ✗ {var}: NOT SET")
            all_good = False

    print("\nOptional variables (need at least one):")
    has_search_api = False
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {'*' * 10} (set)")
            has_search_api = True
        else:
            print(f"  - {var}: not set")

    if not has_search_api:
        print("\n  ⚠ Warning: No search API configured (need TAVILY_API_KEY or PERPLEXITY_API_KEY)")
        all_good = False

    print("\nConfiguration:")
    config_vars = [
        "STAGE1_MODEL", "STAGE1_MAX_LOOPS", "STAGE1_SEARCH_API",
        "STAGE2_MODEL", "STAGE2_STEP_LIMIT", "STAGE2_COST_LIMIT",
        "STAGE3_MODEL"
    ]
    for var in config_vars:
        value = os.getenv(var)
        if value:
            print(f"  • {var}: {value}")
        else:
            print(f"  • {var}: (using default)")

    if all_good:
        print("\n✓ Environment is properly configured!")
    else:
        print("\n✗ Environment has issues. Please check .env file.")

    return all_good


def main():
    """Main entry point for examples."""
    if len(sys.argv) < 2:
        print("Mini Researcher Agent - Usage Examples")
        print("="*70)
        print("\nUsage: python example.py <example_number>")
        print("\nAvailable examples:")
        print("  0 - Check environment configuration")
        print("  1 - Basic usage")
        print("  2 - Custom configuration (intensive)")
        print("  3 - Minimal configuration (quick)")
        print("  4 - Error handling")
        print("  5 - Programmatic workflow (step-by-step)")
        print("  6 - Algorithmic: matrix rectangular region sum")
        print("\nExample:")
        print("  python example.py 1")
        print("  python example.py 0  # Check environment first")
        sys.exit(1)

    example_num = sys.argv[1]

    examples = {
        "0": ("Check Environment", check_environment),
        "1": ("Basic Usage", example_1_basic_usage),
        "2": ("Custom Configuration", example_2_custom_config),
        "3": ("Minimal Configuration", example_3_minimal_config),
        "4": ("Error Handling", example_4_with_error_handling),
        "5": ("Programmatic Workflow", example_5_programmatic_workflow),
        "6": ("Matrix Rectangular Region Sum", example_6_matrix_subregion_sum),
    }

    if example_num not in examples:
        print(f"Error: Unknown example number '{example_num}'")
        print(f"Available: {', '.join(examples.keys())}")
        sys.exit(1)

    title, func = examples[example_num]

    print(f"\n{'='*70}")
    print(f"Running Example {example_num}: {title}")
    print(f"{'='*70}\n")

    func()

    print(f"\n{'='*70}")
    print(f"Example {example_num} Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
