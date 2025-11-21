# MSR-Scientist 🔬

A **minimalist, self-evolving researcher agent** built with [smolagents](https://github.com/huggingface/smolagents) by HuggingFace.

MSR-Scientist is designed to autonomously conduct research following a simple yet powerful evolution cycle:

```
PLAN → IMPLEMENT → VERIFY → SHIP
```

## Philosophy

- **Minimalist**: Simple, functional, no unnecessary complexity
- **Self-Evolving**: Autonomously installs packages and creates tools as needed
- **Research-Focused**: Built specifically for research workflows (hypothesis → experiment → paper)
- **Autonomous**: Makes decisions and evolves without constant human intervention

## Features

### Core Capabilities

✅ **Hypothesis Generation** - Formulate research questions and testable hypotheses
✅ **Experimentation** - Execute code, run experiments with Python/bash
✅ **Autonomous Evolution** - Install packages and create tools dynamically
✅ **Paper Drafting** - Generate LaTeX papers and compile to PDF
✅ **Self-Evolution Loop** - Follow plan → implement → verify → ship cycle

### Built-in Research Tools

- `generate_hypotheses()` - Create structured research hypotheses
- `refine_hypothesis()` - Improve hypotheses based on feedback
- `run_experiment()` - Execute experimental code
- `install_package()` - Autonomously install Python packages
- `create_tool_from_code()` - Create new tools on the fly
- `draft_research_paper()` - Generate and compile LaTeX papers
- `format_results_table()` - Create LaTeX tables from results
- Evolution tracking tools for managing research cycles

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MSR-Scientist.git
cd MSR-Scientist

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Requirements

- Python 3.9+
- HuggingFace API token (for using HF models)
- LaTeX distribution (for PDF compilation) - optional but recommended

## Quick Start

### 1. Set Your HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

### 2. Minimal Example

```python
from msr_scientist import create_research_agent

# Create agent
agent = create_research_agent()

# Give it a research task
agent.research("""
Investigate whether Python list comprehensions are faster than traditional loops.
Run experiments, collect data, and draft a short paper with your findings.
""")
```

That's it! The agent will:
1. ✅ Generate hypotheses
2. ✅ Design and run experiments
3. ✅ Install any needed packages automatically
4. ✅ Create custom tools if necessary
5. ✅ Draft a research paper with results

### 3. Check Outputs

Results are saved in the `outputs/` directory:
- `outputs/experiments/` - Experiment results and data
- `outputs/papers/` - Generated LaTeX and PDF papers
- `outputs/tools/` - Dynamically created tools
- `outputs/evolution/` - Evolution cycle states

## Examples

### Basic Research

```python
from msr_scientist import ResearchAgent

agent = ResearchAgent(verbose=True)

task = """
Research Task: Compare sorting algorithm performance.

1. Generate hypotheses about bubble sort vs quicksort
2. Implement timing experiments
3. Collect results for arrays of size 100, 1000, 10000
4. Draft a paper with your findings
"""

agent.research(task)
```

### Advanced: Self-Evolution in Action

```python
agent = ResearchAgent(verbose=True)

# This task requires packages not initially available
task = """
Analyze text similarity metrics (cosine, Jaccard, Levenshtein).

Requirements:
- Install necessary packages (scikit-learn, python-Levenshtein, etc.)
- Create visualization tools if needed
- Run comparative experiments
- Generate a research paper with tables and analysis

Evolve your capabilities as needed!
"""

agent.research(task)
# The agent will autonomously:
# - Install required packages
# - Create custom plotting tools
# - Conduct experiments
# - Draft a complete paper
```

### Custom Tools

```python
from smolagents import tool
from msr_scientist import ResearchAgent

@tool
def custom_analysis(data: str) -> str:
    """Your domain-specific analysis tool."""
    # Implementation here
    return result

agent = ResearchAgent(
    additional_tools=[custom_analysis],
    verbose=True
)

agent.research("Use custom_analysis to investigate...")
```

## Architecture

```
msr_scientist/
├── agent.py              # Main ResearchAgent with self-evolution prompt
├── tools/
│   ├── hypothesis.py     # Hypothesis generation tools
│   ├── experiment.py     # Experimentation and tool creation
│   ├── paper.py          # LaTeX paper drafting
│   └── evolution.py      # Evolution cycle tracking
└── __init__.py

examples/
├── minimal_example.py    # Simplest usage
├── basic_research.py     # Basic research workflow
├── advanced_research.py  # Self-evolution demonstration
└── custom_tools.py       # Adding custom tools
```

## How Self-Evolution Works

The agent's behavior is guided by a **system prompt** that encodes the self-evolution loop. When the agent encounters a limitation:

1. **Recognize the gap** - "I need package X" or "I need a tool for Y"
2. **Take action** - Install package or create tool autonomously
3. **Continue research** - Use new capability to proceed
4. **Learn** - Reflect and improve for next iteration

This happens naturally through the agent's reasoning, not rigid predefined workflows.

## Configuration

### Using Different Models

```python
from msr_scientist import ResearchAgent

# Use a different HuggingFace model
agent = ResearchAgent(
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    verbose=True
)

# Or provide your own model instance
from smolagents import HfApiModel

custom_model = HfApiModel(
    model_id="your-model",
    token="your-token"
)

agent = ResearchAgent(model=custom_model)
```

### Adding Authorized Imports

```python
agent = ResearchAgent(
    additional_imports=['tensorflow', 'torch', 'transformers'],
    verbose=True
)
```

## Inspiration

MSR-Scientist draws inspiration from:
- [mini-swe-agent](https://github.com/OpenDevin/mini-swe-agent) - Minimalist software engineering agent
- [live-swe-agent](https://github.com/princeton-nlp/SWE-agent) - Software engineering agent framework
- Research automation and self-evolving systems

But focuses specifically on **research workflows** with a **minimalist philosophy**.

## Contributing

Contributions welcome! The goal is to keep this agent minimal yet functional.

Key principles:
- Simplicity over complexity
- Function over features
- Research-focused

## License

Apache-2.0

## Citation

If you use MSR-Scientist in your research:

```bibtex
@software{msr_scientist,
  title={MSR-Scientist: A Minimalist Self-Evolving Research Agent},
  author={MSR-Scientist Contributors},
  year={2025},
  url={https://github.com/your-username/MSR-Scientist}
}
```

## Acknowledgments

- Built with [smolagents](https://github.com/huggingface/smolagents) by HuggingFace
- Uses [PyLaTeX](https://github.com/JelteF/PyLaTeX) for paper generation

---

**Start your research journey:**

```python
from msr_scientist import create_research_agent

agent = create_research_agent()
agent.research("Your research question here...")
```

🔬 **Happy researching!**
