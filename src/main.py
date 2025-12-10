"""researcher.py - Main orchestrator for autonomous research agent

Smolagents-based orchestrator that autonomously coordinates 3 stages:
1. Research & Planning (via research_plan.py)
2. Execute & Analyze (via execute_analyze.py)
3. Report Generation (via report_write.py)

The agent decides when to call each tool based on the research task.
"""

import json
import sys
import uuid
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, LogLevel
from config import get_config, ModelConfig

load_dotenv()

# ============================================================================
# TOOLS - Thin wrappers around stage modules
# ============================================================================

@tool
def research_and_plan(task: str, output_dir: str = "research_outputs") -> str:
    """Conduct deep online research and generate a structured research plan.

    This tool performs web-based research using Tavily or Perplexity, generates
    a comprehensive research plan with hypotheses, methodology, and metrics,
    and extracts citations from sources.

    Args:
        task: The research question or topic to investigate
        output_dir: Directory to save research plan and citations

    Returns:
        JSON string containing paths to generated files and summary
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

    # Run research
    researcher = DeepResearcher(config)
    plan_dict, citations_list = researcher.run_research(task)

    # Add metadata
    plan_dict["task_id"] = task_id
    plan_dict["timestamp"] = datetime.now().isoformat()

    # Save outputs
    plan_file = task_dir / "research_plan.json"
    citations_file = task_dir / "citations.json"

    with open(plan_file, 'w') as f:
        json.dump(plan_dict, f, indent=2)

    with open(citations_file, 'w') as f:
        json.dump(citations_list, f, indent=2)

    # Return result
    result = {
        "success": True,
        "task_id": task_id,
        "task_dir": str(task_dir),
        "plan_file": str(plan_file),
        "citations_file": str(citations_file),
        "num_hypotheses": len(plan_dict.get("hypotheses", [])),
        "num_citations": len(citations_list),
        "methodology": plan_dict.get("methodology", {}).get("approach", "")
    }

    return json.dumps(result, indent=2)


@tool
def execute_and_analyze(plan_file: str, workspace: str | None = None) -> str:
    """Execute the research plan by implementing and running experiments.

    This tool uses mini-swe-agent to convert the research plan into code,
    execute experiments, and collect results.

    Args:
        plan_file: Path to research_plan.json from research_and_plan tool
        workspace: Directory for code execution

    Returns:
        JSON string containing execution results and status
    """
    from execute_analyze import run_implementation, validate_results

    # Auto-generate workspace if not provided
    if workspace is None:
        plan_path = Path(plan_file)
        workspace = str(plan_path.parent / "workspace")

    # Load centralized config
    pipeline_config = get_config()

    # Normalize model name for minisweagent
    model_config = ModelConfig(model_id=pipeline_config.stage2_model.model_id)
    normalized_model = model_config.normalize_model_id()

    results = run_implementation(
        plan_file=plan_file,
        workspace=workspace,
        model_name=normalized_model,
        step_limit=pipeline_config.stage2_step_limit,
        cost_limit=pipeline_config.stage2_cost_limit
    )

    # Validate results
    is_valid, msg = validate_results(results)

    # Save results to workspace
    workspace_path = Path(workspace)
    results_file = workspace_path / "results.json"

    # Prepare results data for saving
    results_data = {
        "task": results.get("task", ""),
        "hypotheses_tested": results.get("hypotheses_tested", []),
        "findings": results.get("findings", []),
        "metrics": results.get("metrics", {}),
        "figures": results.get("figures", []),
        "execution_summary": {
            "exit_status": results.get("exit_status", ""),
            "steps_taken": results.get("message_count", 0),
            "cost_usd": results.get("cost_usd", 0.0)
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Prepare return value
    result = {
        "success": is_valid,
        "validation_message": msg,
        "exit_status": results.get("exit_status", ""),
        "results_file": str(results_file),
        "workspace": workspace,
        "num_findings": len(results.get("findings", [])),
        "num_figures": len(results.get("figures", [])),
        "cost_usd": results.get("cost_usd", 0.0),
        "message_count": results.get("message_count", 0)
    }

    return json.dumps(result, indent=2)


@tool
def generate_report(plan_file: str, citations_file: str, results_file: str, output_dir: str | None = None) -> str:
    """Generate a professional PDF research report with citations and figures.

    This tool synthesizes the research plan, execution results, and citations
    into a comprehensive academic paper in PDF format.

    Args:
        plan_file: Path to research_plan.json
        citations_file: Path to citations.json
        results_file: Path to results.json
        output_dir: Directory for PDF output

    Returns:
        JSON string containing PDF path and status
    """
    from report_write import create_pdf_report, validate_report

    # Auto-generate output_dir if not provided
    if output_dir is None:
        output_dir = str(Path(plan_file).parent)

    # Load centralized config
    pipeline_config = get_config()

    try:
        pdf_path = create_pdf_report(
            plan_file=plan_file,
            citations_file=citations_file,
            results_file=results_file,
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

Your workflow follows 3 stages:

1. **RESEARCH & PLANNING**: Use research_and_plan(task) to conduct deep online research
   - Performs web search with Tavily or Perplexity
   - Generates structured research plan with hypotheses, methodology, metrics
   - Extracts citations from sources
   - Returns paths to research_plan.json and citations.json

2. **EXECUTE & ANALYZE**: Use execute_and_analyze(plan_file) to implement the research
   - Converts research plan to code
   - Executes experiments using mini-swe-agent
   - Collects results and generates visualizations
   - Returns path to results.json

3. **REPORT GENERATION**: Use generate_report(plan_file, citations_file, results_file) to create PDF
   - Synthesizes findings into academic paper
   - Includes citations from research phase
   - Embeds figures from execution phase
   - Returns path to final PDF report

WORKFLOW AUTONOMY:
- You decide when to call each tool based on the task
- Typical flow: research_and_plan → execute_and_analyze → generate_report
- Use outputs from each stage as inputs to the next
- Handle errors gracefully and retry if needed

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
            tools=[research_and_plan, execute_and_analyze, generate_report],
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
                    "open": open,  # Allow file operations for reading research plans and results
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
