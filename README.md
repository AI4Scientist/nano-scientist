# mini-researcher-agent 🔬

**A minimalist self-evolving researcher agent using [smolagents](https://github.com/huggingface/smolagents).**

One file. Three tools. Infinite research potential.

```
PLAN → IMPLEMENT → VERIFY → SHIP
```

## Features

✅ **Hypothesis generation** - Formulate research questions
✅ **Autonomous experimentation** - Execute code, install packages
✅ **Paper drafting** - Generate LaTeX papers
✅ **Self-evolution** - Creates tools and adapts as needed

## Installation

```bash
# Install dependencies with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run install-tectonic

# Or use the quick setup script
./setup.sh
```

**Configure API keys** - Create a `.env` file:
```bash
# Required
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key  # or PERPLEXITY_API_KEY
```

## Quick Start

```bash
python example.py 1  # Basic usage demo
```

The agent will:
1. Generate hypotheses
2. Install any needed packages
3. Run experiments
4. Draft a research paper

## How It Works

The agent has 3 core tools:

- `run_experiment(code)` - Execute Python/bash code
- `install_package(name)` - Install packages autonomously
- `draft_paper(...)` - Generate LaTeX papers

**Self-evolution** is guided by a system prompt that teaches the agent to:
- Install packages when needed
- Write code to create new capabilities
- Follow the research cycle: plan → implement → verify → ship

## Architecture

```
mini-researcher-agent/
├── src/               # Core modules
├── example.py         # Usage examples
├── setup.sh          # Quick setup script
└── requirements.txt  # Dependencies
```

Three-stage pipeline: Planning → Execution → Reporting

## Philosophy

- **Minimal**: Essential tools only, no bloat
- **Autonomous**: Self-evolving through tool creation
- **Research-focused**: Built for the scientific method
- **Stage-based**: Plan → Execute → Report

## Requirements

- Python 3.8+
- OpenAI API key + HuggingFace token
- Tavily or Perplexity API (for search)
- Tectonic (auto-installed for PDF generation)

## License

Apache-2.0

## Inspired By

- [mini-swe-agent](https://github.com/OpenDevin/mini-swe-agent)
- [smolagents](https://github.com/huggingface/smolagents)
