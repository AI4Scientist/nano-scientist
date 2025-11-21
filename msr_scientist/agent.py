"""
Main ResearchAgent implementation using smolagents.

This agent embodies a minimalist self-evolving researcher that can:
- Generate hypotheses
- Run experiments
- Draft papers
- Evolve its own capabilities through a system prompt-guided loop
"""

from smolagents import CodeAgent, HfApiModel, tool
from typing import Optional, List, Dict, Any
import os

# Import all research tools
from msr_scientist.tools.hypothesis import generate_hypotheses, refine_hypothesis
from msr_scientist.tools.experiment import (
    run_experiment,
    create_tool_from_code,
    install_package
)
from msr_scientist.tools.paper import (
    draft_research_paper,
    add_paper_section,
    format_results_table
)
from msr_scientist.tools.evolution import (
    evolution_plan,
    evolution_implement,
    evolution_verify,
    evolution_ship,
    save_evolution_state
)


# System prompt that guides self-evolution behavior
RESEARCHER_SYSTEM_PROMPT = """You are a self-evolving research agent with a minimalist philosophy.

Your core mission is to autonomously conduct research following this evolution cycle:

**PLAN → IMPLEMENT → VERIFY → SHIP**

## Self-Evolution Principles

1. **Autonomy**: When you need a capability, CREATE IT.
   - Missing a Python library? Use install_package() to install it.
   - Need a specialized tool? Use create_tool_from_code() to build it on the fly.
   - Don't wait for permission - evolve your capabilities as needed.

2. **Minimalism**: Keep solutions simple and functional.
   - Prefer straightforward approaches over complex ones.
   - Write clear, readable code.
   - Create tools only when they add real value.

3. **Scientific Rigor**: Follow the research cycle.
   - PLAN: Formulate clear hypotheses and experimental designs
   - IMPLEMENT: Write and execute experimental code
   - VERIFY: Test results, check validity, iterate if needed
   - SHIP: Document findings, draft papers, save artifacts

4. **Continuous Learning**: Each cycle informs the next.
   - Reflect on what worked and what didn't
   - Identify gaps in your capabilities
   - Evolve your toolkit based on real needs
   - Save state for resumption and learning

## Available Core Tools

**Hypothesis Generation:**
- generate_hypotheses(): Create research hypotheses
- refine_hypothesis(): Improve hypotheses based on feedback

**Experimentation:**
- run_experiment(): Execute Python or bash code
- install_package(): Install Python packages autonomously
- create_tool_from_code(): Create new tools dynamically

**Paper Drafting:**
- draft_research_paper(): Generate LaTeX papers and compile to PDF
- add_paper_section(): Update existing papers
- format_results_table(): Create result tables

**Evolution Tracking (Optional):**
- evolution_plan(), evolution_implement(), evolution_verify(), evolution_ship()
- save_evolution_state(): Save your progress

## How to Operate

When given a research task:

1. **Understand** the task and context deeply
2. **Generate hypotheses** about approaches or findings
3. **Plan experiments** to test hypotheses
4. **Implement** by writing code - install libraries or create tools as needed
5. **Execute experiments** and collect results
6. **Verify** results through testing and validation
7. **Draft papers** documenting your findings
8. **Ship** by saving all artifacts and reflecting on improvements

Remember: You're not just using tools - you're evolving your capabilities. If you hit a limitation, overcome it by creating what you need.

Be bold. Be scientific. Be minimal. Evolve."""


class ResearchAgent:
    """
    A minimalist self-evolving research agent built on smolagents.

    This agent autonomously conducts research by following a simple cycle:
    plan → implement → verify → ship, evolving its capabilities as needed.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        hf_token: Optional[str] = None,
        additional_tools: Optional[List] = None,
        additional_imports: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Initialize the ResearchAgent.

        Args:
            model: Pre-configured model instance (if None, creates HfApiModel)
            model_id: Model ID for HuggingFace API (default: Qwen2.5-Coder-32B-Instruct)
            hf_token: HuggingFace API token (or set HF_TOKEN env var)
            additional_tools: Extra tools beyond the built-in research tools
            additional_imports: Additional Python modules to authorize for code execution
            verbose: Enable verbose output
        """
        # Setup model
        if model is None:
            token = hf_token or os.getenv("HF_TOKEN")
            if not token:
                raise ValueError(
                    "HuggingFace token required. Set HF_TOKEN env var or pass hf_token parameter."
                )
            model = HfApiModel(model_id=model_id, token=token)

        # Collect all research tools
        research_tools = [
            # Hypothesis tools
            generate_hypotheses,
            refine_hypothesis,
            # Experiment tools
            run_experiment,
            create_tool_from_code,
            install_package,
            # Paper tools
            draft_research_paper,
            add_paper_section,
            format_results_table,
            # Evolution tools
            evolution_plan,
            evolution_implement,
            evolution_verify,
            evolution_ship,
            save_evolution_state,
        ]

        # Add any additional tools
        if additional_tools:
            research_tools.extend(additional_tools)

        # Default additional imports for research work
        default_imports = [
            'numpy',
            'pandas',
            'matplotlib',
            'scipy',
            'sklearn',
            'json',
            'requests',
            'pathlib',
            'subprocess',
            'tempfile',
        ]

        if additional_imports:
            default_imports.extend(additional_imports)

        # Create the CodeAgent with research tools and system prompt
        self.agent = CodeAgent(
            tools=research_tools,
            model=model,
            additional_authorized_imports=default_imports,
            add_base_tools=True,  # Include web search, file operations, etc.
            verbose=verbose,
            system_prompt=RESEARCHER_SYSTEM_PROMPT
        )

        self.verbose = verbose

    def research(self, task: str, **kwargs) -> Any:
        """
        Conduct research on a given task.

        The agent will autonomously:
        - Generate hypotheses
        - Design and run experiments
        - Install any needed packages
        - Create custom tools if needed
        - Draft research papers
        - Follow the plan → implement → verify → ship cycle

        Args:
            task: Research task description with context
            **kwargs: Additional arguments passed to agent.run()

        Returns:
            The agent's research results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 MSR-Scientist Research Agent")
            print(f"{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")

        result = self.agent.run(task, **kwargs)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ Research Complete")
            print(f"{'='*60}\n")

        return result

    def __call__(self, task: str, **kwargs) -> Any:
        """Allow using the agent as a callable."""
        return self.research(task, **kwargs)

    @property
    def tools(self):
        """Access the agent's tools."""
        return self.agent.tools

    def add_tool(self, tool):
        """
        Add a new tool to the agent's toolkit.

        Args:
            tool: A tool function decorated with @tool or a Tool instance
        """
        self.agent.tools.append(tool)

    def get_agent(self):
        """Get the underlying CodeAgent instance."""
        return self.agent


# Convenience function for quick usage
def create_research_agent(
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    hf_token: Optional[str] = None,
    **kwargs
) -> ResearchAgent:
    """
    Create a ResearchAgent with sensible defaults.

    Args:
        model_id: HuggingFace model ID
        hf_token: HuggingFace API token
        **kwargs: Additional arguments for ResearchAgent

    Returns:
        Configured ResearchAgent instance
    """
    return ResearchAgent(model_id=model_id, hf_token=hf_token, **kwargs)
