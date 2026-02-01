"""researcher.py - Main orchestrator for autonomous research agent

Smolagents-based orchestrator that autonomously coordinates 3 stages:
1. Research & Planning (via research_plan.py) → research_proposal.md
2. Execute & Analyze (via execute_analyze.py) → workspace/ + README.md
3. Report Generation (via report_write.py) → report/main.pdf

The agent decides when to call each tool based on the research task.
"""

import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, LogLevel
from src.config import get_config, ModelConfig

load_dotenv()

# ============================================================================
# TOOLS - Thin wrappers around stage modules
# ============================================================================

@tool
def research_and_plan(task: str, output_dir: str = "research_outputs") -> str:
    """Conduct deep online research and generate a research proposal.

    This tool performs web-based research using Tavily or Perplexity, generates
    a comprehensive research proposal with hypotheses, methodology, metrics,
    and citations in markdown format.

    Args:
        task: The research question or topic to investigate
        output_dir: Directory to save research_proposal.md

    Returns:
        JSON string containing path to research_proposal.md and summary
    """
    from research_plan import DeepResearcher, Configuration, SearchAPI

    # Create task-specific output directory
    task_id = str(uuid.uuid4())[:8]
    task_dir = Path(output_dir) / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Load centralized config
    pipeline_config = get_config()

    # Configure researcher
    config = Configuration(
        max_web_research_loops=pipeline_config.stage1_max_loops,
        model_id=pipeline_config.stage1_model.model_id,
        search_api=SearchAPI(pipeline_config.stage1_search_api),
        temperature=pipeline_config.stage1_model.temperature
    )

    # Run research - returns markdown content
    researcher = DeepResearcher(config)
    proposal_md = researcher.run_research(task, task_id)

    # Save proposal markdown
    proposal_file = task_dir / "research_proposal.md"
    proposal_file.write_text(proposal_md, encoding='utf-8')

    # Count hypotheses and citations from markdown
    import re
    hypotheses = re.findall(r'^\d+\.\s+.+$', proposal_md, re.MULTILINE)
    citations = re.findall(r'-\s+\*\*ref\d+\*\*:', proposal_md)

    # Return result
    result = {
        "success": True,
        "task_id": task_id,
        "task_dir": str(task_dir),
        "proposal_file": str(proposal_file),
        "num_hypotheses": len(hypotheses),
        "num_citations": len(citations)
    }

    return json.dumps(result, indent=2)


@tool
def execute_and_analyze(proposal_file: str, workspace: str | None = None) -> str:
    """Execute the research plan by implementing and running experiments.

    This tool reads the research proposal, sets up a Docker-ready workspace,
    uses mini-swe-agent to implement experiments, and generates a README
    describing the artifacts.

    Args:
        proposal_file: Path to research_proposal.md from research_and_plan tool
        workspace: Directory for code execution (auto-generated if not provided)

    Returns:
        JSON string containing execution results and workspace path
    """
    from execute_analyze import run_implementation, validate_results

    # Auto-generate workspace if not provided
    if workspace is None:
        proposal_path = Path(proposal_file)
        workspace = str(proposal_path.parent / "workspace")

    # Load centralized config
    pipeline_config = get_config()

    # Normalize model name for minisweagent
    model_config = ModelConfig(model_id=pipeline_config.stage2_model.model_id)
    normalized_model = model_config.normalize_model_id()

    results = run_implementation(
        proposal_file=proposal_file,
        workspace=workspace,
        model_name=normalized_model,
        step_limit=pipeline_config.stage2_step_limit,
        cost_limit=pipeline_config.stage2_cost_limit
    )

    # Validate results
    is_valid, msg = validate_results(results)

    # Prepare return value
    result = {
        "success": is_valid,
        "validation_message": msg,
        "exit_status": results.get("exit_status", ""),
        "workspace": workspace,
        "readme_path": results.get("readme_path", ""),
        "artifacts": results.get("artifacts", {}),
        "cost_usd": results.get("cost_usd", 0.0),
        "message_count": results.get("message_count", 0)
    }

    return json.dumps(result, indent=2)


@tool
def report_and_write(proposal_file: str, workspace: str, output_dir: str | None = None) -> str:
    """Generate a professional PDF research report using ACM template.

    This tool reads the research proposal and workspace artifacts to generate
    a comprehensive academic paper in PDF format using the ACM Conference template.

    Args:
        proposal_file: Path to research_proposal.md
        workspace: Path to workspace directory with artifacts
        output_dir: Directory for PDF output (auto-generated if not provided)

    Returns:
        JSON string containing PDF path and status
    """
    from report_write import create_pdf_report, validate_report

    # Auto-generate output_dir if not provided
    if output_dir is None:
        output_dir = str(Path(proposal_file).parent)

    # Load centralized config
    pipeline_config = get_config()

    try:
        pdf_path = create_pdf_report(
            proposal_file=proposal_file,
            workspace_path=workspace,
            output_dir=output_dir,
            model_id=pipeline_config.stage3_model.model_id
        )
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "pdf_path": None
        }, indent=2)

    # Validate report
    is_valid, msg = validate_report(pdf_path)

    result = {
        "success": is_valid,
        "validation_message": msg,
        "pdf_path": pdf_path,
        "output_dir": output_dir
    }

    return json.dumps(result, indent=2)


# ============================================================================
# SYSTEM PROMPT - Guides agent through 3-stage workflow
# ============================================================================

SYSTEM_PROMPT = """You are an autonomous research agent that conducts end-to-end research projects.

Your workflow follows 3 stages, all using markdown-based files:

1. **RESEARCH & PLANNING**: Use research_and_plan(task) to conduct deep online research
   - Performs web search with Tavily or Perplexity
   - Generates research_proposal.md with:
     - Hypotheses, methodology, metrics
     - Citations in [@ref] format
     - Background context from literature
   - Returns path to research_proposal.md

2. **EXECUTE & ANALYZE**: Use execute_and_analyze(proposal_file) to implement the research
   - Reads research_proposal.md for context
   - Sets up Docker-ready workspace (Dockerfile, requirements.txt, main.py)
   - Executes experiments using mini-swe-agent
   - Generates workspace/README.md describing artifacts
   - Returns workspace path

3. **REPORT GENERATION**: Use report_and_write(proposal_file, workspace) to create PDF
   - Reads research_proposal.md for content and citations
   - Reads workspace/README.md for artifact information
   - Uses ACM Conference template for professional formatting
   - Converts [@ref] citations to LaTeX \\cite{ref}
   - Returns path to final PDF report

WORKFLOW AUTONOMY:
- You decide when to call each tool based on the task
- Typical flow: research_and_plan → execute_and_analyze → report_and_write
- Pass proposal_file and workspace paths between stages
- Handle errors gracefully and retry if needed

OUTPUT FILES:
- research_proposal.md: Single source of truth with plan + citations
- workspace/: Contains code, figures, data, README.md
- report/: Contains ACM template with main.tex, reference.bib, main.pdf

Be thorough, systematic, and ensure all three stages complete successfully.
"""


# ============================================================================
# AGENT CLASS - Smolagents CodeAgent wrapper
# ============================================================================

class ResearchAgent:
    """Autonomous research agent powered by smolagents."""

    def __init__(self, model_id: str | None = None, api_key: str | None = None, verbose: bool = True):
        """Initialize the research agent.

        Args:
            model_id: Model ID for the agent LLM
            api_key: API key (optional, auto-detected by LiteLLM based on model provider)
            verbose: Enable verbose logging
        """
        # Load centralized config
        pipeline_config = get_config()

        # Use provided model_id or get from config
        if model_id:
            temp_config = ModelConfig(model_id=model_id)
            normalized_model = temp_config.normalize_model_id()
        else:
            normalized_model = pipeline_config.agent_model.normalize_model_id()

        # LiteLLM auto-detects API keys based on model prefix
        # Only pass api_key if explicitly provided
        model_kwargs = {'model_id': normalized_model}
        if api_key:
            model_kwargs['api_key'] = api_key

        self.agent = CodeAgent(
            tools=[research_and_plan, execute_and_analyze, report_and_write],
            model=LiteLLMModel(**model_kwargs),
            add_base_tools=True,
            verbosity_level=LogLevel.INFO if verbose else LogLevel.OFF,
            instructions=SYSTEM_PROMPT,
            additional_authorized_imports=[
                "unittest",
                "json",
                "os",
                "sys",
                "pathlib",
                "typing",
                "dataclasses",
                "enum",
                "abc",
                "copy",
                "functools",
                "operator",
                "pickle",
                "shelve",
                "csv",
                "hashlib",
                "hmac",
                "secrets",
                "struct",
                "codecs",
                "io",
                "argparse",
                "logging",
                "warnings",
                "contextlib",
                "tempfile",
                "shutil",
                "glob",
                "fnmatch",
                "decimal",
                "fractions",
                "numbers",
                "cmath",
                "array",
                "bisect",
                "heapq",
                "string",
                "textwrap",
                "pprint",
                "difflib",
                "traceback",
                "inspect",
                "dis",
                "ast",
                "timeit",
                "gc",
                "resource",
                "platform",
                "errno",
                "curses",
                "getpass",
                "getopt",
                "zipfile",
                "tarfile",
                "gzip",
                "bz2",
                "lzma",
                "zlib",
                "base64",
                "binascii",
                "quopri",
                "uu",
                "html",
                "xml",
                "webbrowser",
                "uuid",
                "socketserver",
                "urllib",
                "http",
                "ftplib",
                "poplib",
                "imaplib",
                "nntplib",
                "smtplib",
                "smtpd",
                "telnetlib",
                "ssl",
                "email",
                "mailbox",
                "mimetypes",
                "wave",
                "colorsys",
                "imghdr",
                "sndhdr",
                "aifc",
                "sunau",
                "chunk",
                "pwd",
                "spwd",
                "grp",
                "crypt",
                "termios",
                "tty",
                "pty",
                "fcntl",
                "pipes",
                "signal",
                "mmap",
                "ctypes",
                "subprocess",
                "threading",
                "multiprocessing",
                "concurrent",
                "sched",
                "asyncio",
                "socket",
                "select",
                "selectors",
                "asynchat",
                "asyncore",
                "token",
                "keyword",
                "tokenize",
                "tabnanny",
                "pyclbr",
                "py_compile",
                "compileall",
                "pydoc",
                "doctest",
                "test",
                "bdb",
                "faulthandler",
                "pdb",
                "profile",
                "pstats",
                "dbm",
                "sqlite3",
                "zoneinfo",
                "locale",
                "gettext",
                "cmd",
                "shlex",
                "configparser",
                "fileinput",
                "readline",
                "rlcompleter",
            ],
            executor_kwargs={
                "additional_functions": {
                    "open": open,  # Allow file operations for reading research proposals
                }
            }
        )
        self.verbose = verbose

    def __call__(self, task: str, **kwargs):
        """Execute a research task.

        Args:
            task: Research question or topic
            **kwargs: Additional arguments passed to agent.run()

        Returns:
            Agent execution result
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔬 Starting Research: {task}")
            print(f"{'='*70}\n")

        result = self.agent.run(task, **kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ Research Complete")
            print(f"{'='*70}\n")

        return result


# ============================================================================
# CLI - Main entry point
# ============================================================================

def main():
    """CLI entry point for the research agent."""
    if len(sys.argv) < 2:
        print("Usage: python researcher.py 'research task'")
        print("\nExample:")
        print("  python researcher.py 'Compare sorting algorithms and analyze performance'")
        print("\nEnvironment variables:")
        print("  AGENT_MODEL - Main orchestrator")
        print("  STAGE1_MODEL - Research phase")
        print("  STAGE1_MAX_LOOPS - Research iterations")
        print("  STAGE1_SEARCH_API - Search API")
        print("  STAGE2_MODEL - Execution phase")
        print("  STAGE2_STEP_LIMIT - Max coding steps")
        print("  STAGE2_COST_LIMIT - Max cost in USD")
        print("  STAGE3_MODEL - Report phase")
        print("")
        print("  API Keys (auto-detected by LiteLLM based on model provider):")
        print("  OPENAI_API_KEY - For OpenAI models (gpt-*)")
        print("  ANTHROPIC_API_KEY - For Anthropic models (claude-*)")
        print("  HF_TOKEN - For HuggingFace models")
        print("  TAVILY_API_KEY or PERPLEXITY_API_KEY - For web search")
        sys.exit(1)

    task = " ".join(sys.argv[1:])

    # Initialize and run agent
    agent = ResearchAgent()
    result = agent(task)

    print(f"\n📋 Final Result:\n{result}\n")


if __name__ == "__main__":
    main()
