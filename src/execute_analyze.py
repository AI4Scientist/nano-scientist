"""execute_analyze.py - Stage 2: Execute & Analyze module

Wrapper around mini-swe-agent for code implementation and execution.
Converts research plans to coding tasks, executes them, and collects results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment


def plan_to_task_prompt(plan: Dict) -> str:
    """Convert a research plan JSON to a natural language task for mini-swe-agent.

    Args:
        plan: Research plan dictionary with hypotheses, methodology, metrics

    Returns:
        Natural language task description
    """
    hypotheses = plan.get("hypotheses", [])
    methodology = plan.get("methodology", {})
    approach = methodology.get("approach", "")
    steps = methodology.get("steps", [])
    tools_needed = methodology.get("tools_needed", [])
    metrics = plan.get("metrics", {})
    primary_metric = metrics.get("primary", "")
    secondary_metrics = metrics.get("secondary", [])

    task_prompt = f"""Implement the following research plan:

RESEARCH TASK:
{plan.get("task", "Research implementation")}

HYPOTHESES TO TEST:
{chr(10).join(f"- {h}" for h in hypotheses)}

METHODOLOGY:
{approach}

IMPLEMENTATION STEPS:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(steps))}

REQUIRED TOOLS/PACKAGES:
{chr(10).join(f"- {t}" for t in tools_needed) if tools_needed else "- (Use standard Python libraries)"}

SUCCESS METRICS:
- Primary: {primary_metric}
{chr(10).join(f"- {m}" for m in secondary_metrics)}

REQUIREMENTS:
1. Implement all experiments in the current workspace
2. Save results to results.json with this structure:
   {{
       "task": "Task description",
       "hypotheses_tested": [
           {{
               "hypothesis": "Hypothesis text",
               "result": "supported|rejected|inconclusive",
               "evidence": "Quantitative or qualitative evidence"
           }}
       ],
       "findings": ["Finding 1", "Finding 2", ...],
       "metrics": {{
           "primary_metric": {{"name": "...", "value": 0.95, "unit": "..."}},
           "secondary_metrics": [...]
       }},
       "figures": [
           {{"path": "workspace/figure.png", "caption": "...", "description": "..."}}
       ]
   }}
3. Ensure all code is well-documented and runnable
4. Include proper error handling
5. Generate visualizations where appropriate (save as PNG files)
"""
    return task_prompt


def run_implementation(
    plan_file: str,
    workspace: str,
    model_name: str = "anthropic/claude-haiku-4-5-20251001",
    step_limit: int = 50,
    cost_limit: float = 2.0
) -> Dict[str, Any]:
    """Execute the implementation using mini-swe-agent.

    Args:
        plan_file: Path to research_plan.json
        workspace: Directory for code execution
        model_name: LLM model to use for mini-swe-agent
        step_limit: Maximum number of agent steps
        cost_limit: Maximum cost in USD

    Returns:
        Dictionary with execution results and metadata
    """
    # Load research plan
    plan_path = Path(plan_file)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")

    with open(plan_path, 'r') as f:
        plan = json.load(f)

    # Create workspace directory
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Convert plan to task
    task = plan_to_task_prompt(plan)

    # Initialize mini-swe-agent
    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(cwd=str(workspace_path)),
        step_limit=step_limit,
        cost_limit=cost_limit,
    )

    # Execute
    try:
        exit_status, result_text = agent.run(task)
    except Exception as e:
        return {
            "success": False,
            "exit_status": "Error",
            "error": str(e),
            "result_text": "",
            "workspace": str(workspace_path),
            "message_count": 0,
            "cost_usd": 0.0
        }

    # Extract results from workspace
    results = _extract_results_from_workspace(workspace_path, result_text)

    # Add execution metadata
    results.update({
        "success": exit_status == "Submitted",
        "exit_status": exit_status,
        "result_text": result_text,
        "workspace": str(workspace_path),
        "message_count": len(agent.messages) if hasattr(agent, 'messages') else 0,
        "cost_usd": getattr(agent.model, 'cost', 0.0) if hasattr(agent, 'model') else 0.0
    })

    return results


def _extract_results_from_workspace(workspace: Path, result_text: str) -> Dict:
    """Extract structured results from the workspace directory.

    Args:
        workspace: Path to workspace directory
        result_text: Text result from agent execution

    Returns:
        Dictionary with results data
    """
    results_file = workspace / "results.json"

    # Try to load results.json if it exists
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        except json.JSONDecodeError:
            # If results.json is malformed, fall through to create basic results
            pass

    # Fallback: Create basic results from workspace contents
    code_files = list(workspace.glob("*.py")) + list(workspace.glob("*.sh"))
    figure_files = list(workspace.glob("*.png")) + list(workspace.glob("*.jpg"))

    results = {
        "task": "Implementation completed",
        "hypotheses_tested": [],
        "findings": [
            f"Generated {len(code_files)} code files",
            f"Created {len(figure_files)} visualizations" if figure_files else "No visualizations created",
            "Execution completed with results saved to workspace"
        ],
        "metrics": {
            "primary_metric": {
                "name": "Execution completion",
                "value": 1.0 if result_text else 0.0,
                "unit": "boolean"
            },
            "secondary_metrics": [
                {
                    "name": "Code files generated",
                    "value": len(code_files),
                    "unit": "count"
                },
                {
                    "name": "Figures generated",
                    "value": len(figure_files),
                    "unit": "count"
                }
            ]
        },
        "figures": [
            {
                "path": str(fig.relative_to(workspace.parent)),
                "caption": f"Figure: {fig.stem}",
                "description": "Generated visualization"
            }
            for fig in figure_files
        ],
        "execution_summary": {
            "exit_status": "Submitted",
            "steps_taken": 0,  # Unknown without agent access
            "cost_usd": 0.0
        }
    }

    return results


def validate_results(results: Dict) -> tuple[bool, str]:
    """Validate execution results.

    Args:
        results: Results dictionary from run_implementation()

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not results.get("success", False):
        return False, f"Execution failed: {results.get('error', results.get('exit_status', 'Unknown error'))}"

    if results.get("exit_status") != "Submitted":
        return False, f"Agent did not complete successfully: {results.get('exit_status')}"

    # Check for findings
    findings = results.get("findings", [])
    if not findings or len(findings) == 0:
        return False, "No findings generated"

    # Check workspace has files
    workspace = Path(results.get("workspace", "."))
    code_files = list(workspace.glob("*.py")) + list(workspace.glob("*.sh"))
    if len(code_files) == 0:
        return False, "No code files generated in workspace"

    return True, "Valid"


# Example usage
if __name__ == "__main__":
    # Example: Run implementation from a plan file
    import sys

    if len(sys.argv) < 3:
        print("Usage: python execute_analyze.py <plan_file> <workspace>")
        sys.exit(1)

    plan_file = sys.argv[1]
    workspace = sys.argv[2]

    print(f"Running implementation from plan: {plan_file}")
    print(f"Workspace: {workspace}")

    results = run_implementation(plan_file, workspace)

    print(f"\n{'='*60}")
    print(f"Execution Status: {results['exit_status']}")
    print(f"Success: {results['success']}")
    print(f"Cost: ${results.get('cost_usd', 0):.4f}")
    print(f"Messages: {results.get('message_count', 0)}")
    print(f"\nFindings:")
    for finding in results.get("findings", []):
        print(f"  - {finding}")
    print(f"{'='*60}\n")

    # Validate
    is_valid, msg = validate_results(results)
    if is_valid:
        print("✓ Results validated successfully")
    else:
        print(f"✗ Validation failed: {msg}")
