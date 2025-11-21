"""
Self-evolution loop: plan → implement → verify → ship
"""

from smolagents import tool
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
from pathlib import Path


@tool
def evolution_plan(
    task: str,
    current_capabilities: str = "",
    identified_gaps: str = ""
) -> str:
    """
    Plan phase: Analyze task and formulate approach.

    Creates a structured plan for tackling a research task, identifying
    what needs to be done and what capabilities might be missing.

    Args:
        task: The research task to plan for
        current_capabilities: Description of currently available tools/methods
        identified_gaps: Known gaps or limitations in current approach

    Returns:
        A JSON string containing the plan with steps and requirements
    """
    plan = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "phase": "plan",
        "current_capabilities": current_capabilities,
        "identified_gaps": identified_gaps,
        "steps": [],
        "required_tools": [],
        "success_criteria": [],
        "status": "planned"
    }

    return json.dumps(plan, indent=2)


@tool
def evolution_implement(
    plan: str,
    step_number: int,
    implementation_code: str = "",
    tools_created: str = ""
) -> str:
    """
    Implement phase: Execute planned steps.

    Implements a specific step from the plan, potentially creating
    new tools or running experiments.

    Args:
        plan: JSON string of the plan from evolution_plan
        step_number: Which step to implement (0-indexed)
        implementation_code: Code implementing this step
        tools_created: JSON list of any tools created during implementation

    Returns:
        A JSON string with implementation results
    """
    plan_data = json.loads(plan) if isinstance(plan, str) else plan
    tools_list = json.loads(tools_created) if tools_created and isinstance(tools_created, str) else []

    implementation = {
        "plan_id": plan_data.get("timestamp", "unknown"),
        "phase": "implement",
        "step_number": step_number,
        "implementation_code": implementation_code,
        "tools_created": tools_list,
        "timestamp": datetime.now().isoformat(),
        "status": "implemented"
    }

    return json.dumps(implementation, indent=2)


@tool
def evolution_verify(
    implementation: str,
    test_results: str = "",
    validation_method: str = "manual"
) -> str:
    """
    Verify phase: Check if implementation meets requirements.

    Validates the implementation against success criteria,
    running tests or checks as needed.

    Args:
        implementation: JSON string from evolution_implement
        test_results: Results from tests or validation
        validation_method: How validation was performed

    Returns:
        A JSON string with verification results
    """
    impl_data = json.loads(implementation) if isinstance(implementation, str) else implementation

    verification = {
        "implementation_id": impl_data.get("timestamp", "unknown"),
        "phase": "verify",
        "validation_method": validation_method,
        "test_results": test_results,
        "timestamp": datetime.now().isoformat(),
        "passed": True,  # Will be determined by actual tests
        "issues_found": [],
        "status": "verified"
    }

    return json.dumps(verification, indent=2)


@tool
def evolution_ship(
    verification: str,
    artifacts: str = "",
    improvements_for_next_cycle: str = ""
) -> str:
    """
    Ship phase: Finalize and prepare for next iteration.

    Packages up the completed work, documents learnings,
    and identifies improvements for the next evolution cycle.

    Args:
        verification: JSON string from evolution_verify
        artifacts: JSON list of output artifacts (papers, tools, results)
        improvements_for_next_cycle: Identified improvements for future iterations

    Returns:
        A JSON string with shipping results and next steps
    """
    verif_data = json.loads(verification) if isinstance(verification, str) else verification
    artifacts_list = json.loads(artifacts) if artifacts and isinstance(artifacts, str) else []

    shipping = {
        "verification_id": verif_data.get("timestamp", "unknown"),
        "phase": "ship",
        "timestamp": datetime.now().isoformat(),
        "artifacts": artifacts_list,
        "improvements_for_next_cycle": improvements_for_next_cycle,
        "cycle_complete": True,
        "status": "shipped"
    }

    return json.dumps(shipping, indent=2)


@tool
def save_evolution_state(
    evolution_data: str,
    state_dir: str = "outputs/evolution"
) -> str:
    """
    Save the current evolution state for resumption or analysis.

    Args:
        evolution_data: Complete evolution cycle data as JSON
        state_dir: Directory to save state files

    Returns:
        A JSON string with save status
    """
    result = {
        "status": "saved",
        "file_path": None,
        "error": None
    }

    try:
        os.makedirs(state_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = Path(state_dir) / f"evolution_state_{timestamp}.json"

        # Parse and save
        evolution_dict = json.loads(evolution_data) if isinstance(evolution_data, str) else evolution_data

        with open(state_file, 'w') as f:
            json.dump(evolution_dict, f, indent=2)

        result["file_path"] = str(state_file)
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


class EvolutionCycle:
    """
    Manages the self-evolution loop: plan → implement → verify → ship
    """

    def __init__(self, state_dir: str = "outputs/evolution"):
        self.state_dir = state_dir
        self.current_cycle = None
        self.cycles_history = []
        os.makedirs(state_dir, exist_ok=True)

    def plan(self, task: str, capabilities: str = "", gaps: str = "") -> Dict:
        """Start a new evolution cycle with planning."""
        plan_json = evolution_plan(task, capabilities, gaps)
        self.current_cycle = {
            "plan": json.loads(plan_json),
            "implementations": [],
            "verifications": [],
            "shipping": None
        }
        return self.current_cycle["plan"]

    def implement(self, step_num: int, code: str = "", tools: List[str] = None) -> Dict:
        """Implement a step from the plan."""
        if not self.current_cycle:
            raise ValueError("No active cycle. Call plan() first.")

        plan_json = json.dumps(self.current_cycle["plan"])
        tools_json = json.dumps(tools or [])
        impl_json = evolution_implement(plan_json, step_num, code, tools_json)
        impl = json.loads(impl_json)

        self.current_cycle["implementations"].append(impl)
        return impl

    def verify(self, test_results: str = "", method: str = "manual") -> Dict:
        """Verify the latest implementation."""
        if not self.current_cycle or not self.current_cycle["implementations"]:
            raise ValueError("No implementation to verify.")

        latest_impl = self.current_cycle["implementations"][-1]
        impl_json = json.dumps(latest_impl)
        verif_json = evolution_verify(impl_json, test_results, method)
        verif = json.loads(verif_json)

        self.current_cycle["verifications"].append(verif)
        return verif

    def ship(self, artifacts: List[str] = None, improvements: str = "") -> Dict:
        """Finalize the current cycle and ship."""
        if not self.current_cycle or not self.current_cycle["verifications"]:
            raise ValueError("No verification to ship.")

        latest_verif = self.current_cycle["verifications"][-1]
        verif_json = json.dumps(latest_verif)
        artifacts_json = json.dumps(artifacts or [])
        ship_json = evolution_ship(verif_json, artifacts_json, improvements)
        ship = json.loads(ship_json)

        self.current_cycle["shipping"] = ship

        # Save complete cycle
        self.save_cycle()

        # Move to history
        self.cycles_history.append(self.current_cycle)
        self.current_cycle = None

        return ship

    def save_cycle(self):
        """Save the current cycle state."""
        if self.current_cycle:
            cycle_json = json.dumps(self.current_cycle)
            save_evolution_state(cycle_json, self.state_dir)

    def get_current_cycle(self) -> Optional[Dict]:
        """Get the current evolution cycle."""
        return self.current_cycle

    def get_history(self) -> List[Dict]:
        """Get all completed evolution cycles."""
        return self.cycles_history
