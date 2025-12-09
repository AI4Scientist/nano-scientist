"""mini-researcher-agent: A minimalist autonomous AI researcher (single file)"""

import subprocess, json, os, sys, tempfile
from pathlib import Path
from pylatex import Document, Section, Command
from pylatex.utils import NoEscape
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, tool, LogLevel


# ============================================================================
# TOOLS
# ============================================================================

@tool
def run_experiment(code: str, language: str = "python") -> str:
    """Execute code (python/bash) and return results as JSON.

    Args:
        code: Code to execute
        language: "python" or "bash"
    """
    try:
        if language == "python":
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tf = f.name
            try:
                p = subprocess.run([sys.executable, tf], capture_output=True, text=True, timeout=30)
            finally:
                os.unlink(tf)
        else:
            p = subprocess.run(code, shell=True, capture_output=True, text=True, timeout=30)
        return json.dumps({"stdout": p.stdout, "stderr": p.stderr, "rc": p.returncode})
    except Exception as e:
        return json.dumps({"stdout": "", "stderr": str(e), "rc": -1})


@tool
def install_package(package_name: str) -> str:
    """Install Python package via pip.

    Args:
        package_name: Package to install
    """
    try:
        p = subprocess.run([sys.executable, "-m", "pip", "install", package_name],
                          capture_output=True, text=True, timeout=120)
        return f"✓ {package_name}" if p.returncode == 0 else f"✗ {p.stderr}"
    except Exception as e:
        return f"✗ {e}"


@tool
def draft_paper(title: str, authors: str, abstract: str, sections: str, output_dir: str = "research_outputs") -> str:
    """Generate LaTeX research paper.

    Args:
        title: Paper title
        authors: Authors
        abstract: Abstract text
        sections: JSON list: [{"title":"...","content":"..."}]
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    secs = json.loads(sections) if isinstance(sections, str) else sections

    doc = Document()
    doc.preamble.append(Command('title', title))
    doc.preamble.append(Command('author', authors))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    with doc.create(Section('Abstract', numbering=False)):
        doc.append(abstract)
    for s in secs:
        with doc.create(Section(s.get('title', 'Untitled'))):
            doc.append(s.get('content', ''))

    safe = "".join(c if c.isalnum() else "_" for c in title)[:50]
    path = Path(output_dir) / f"paper_{safe}"
    doc.generate_tex(str(path))

    # Try PDF (fail silently if LaTeX unavailable)
    try:
        doc.generate_pdf(str(path), clean_tex=False)
        return f"📄 {path}.tex\n📕 {path}.pdf"
    except:
        return f"📄 {path}.tex (PDF failed - LaTeX not available)"


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """Autonomous research agent: PLAN → IMPLEMENT → VERIFY → SHIP

Principles:
- Autonomy: Need package? Install it. Need tool? Code it.
- Scientific: Hypothesize, experiment, verify, document.

Tools: run_experiment(code,language), install_package(name), draft_paper(...)

Workflow: Plan → Code → Test → Document. Be bold, evolve."""


# ============================================================================
# AGENT
# ============================================================================

class ResearchAgent:
    """Minimalist autonomous research agent"""

    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507", hf_token=None, verbose=True):
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("Set HF_TOKEN in .env or pass hf_token")

        self.agent = CodeAgent(
            tools=[run_experiment, install_package, draft_paper],
            model=InferenceClientModel(model_id=model_id, token=token),
            add_base_tools=True,
            verbosity_level=LogLevel.INFO if verbose else LogLevel.OFF,
            instructions=SYSTEM_PROMPT
        )
        self.verbose = verbose

    def __call__(self, task: str, **kwargs):
        if self.verbose:
            print(f"\n{'='*60}\n🔬 Research: {task}\n{'='*60}\n")
        result = self.agent.run(task, **kwargs)
        if self.verbose:
            print(f"\n{'='*60}\n✅ Complete\n{'='*60}\n")
        return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python mini_researcher_agent.py 'research task'")
        sys.exit(1)
    ResearchAgent()(" ".join(sys.argv[1:]))
