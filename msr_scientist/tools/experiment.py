"""
Experimentation tools for running code and creating dynamic tools.
"""

from smolagents import tool, Tool
from typing import Dict, Any, Optional
import subprocess
import json
import os
import sys
import tempfile
from pathlib import Path


@tool
def run_experiment(
    code: str,
    language: str = "python",
    save_output: bool = True,
    output_dir: str = "outputs/experiments"
) -> str:
    """
    Execute experimental code and return the results.

    This tool runs code in a controlled environment and captures output.
    Supports Python code and shell commands.

    Args:
        code: The code to execute
        language: Programming language ("python" or "bash")
        save_output: Whether to save output to a file
        output_dir: Directory to save outputs

    Returns:
        A JSON string with execution results including stdout, stderr, and return code
    """
    result = {
        "code": code,
        "language": language,
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        "output_file": None
    }

    try:
        # Create output directory if needed
        if save_output:
            os.makedirs(output_dir, exist_ok=True)

        # Execute code based on language
        if language == "python":
            # Create a temporary file for Python code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                process = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                result["stdout"] = process.stdout
                result["stderr"] = process.stderr
                result["return_code"] = process.returncode
            finally:
                os.unlink(temp_file)

        elif language == "bash":
            process = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
            result["return_code"] = process.returncode

        else:
            result["stderr"] = f"Unsupported language: {language}"
            result["return_code"] = 1

        # Save output if requested
        if save_output and result["return_code"] == 0:
            import hashlib
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            output_file = Path(output_dir) / f"experiment_{code_hash}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            result["output_file"] = str(output_file)

    except subprocess.TimeoutExpired:
        result["stderr"] = "Execution timed out (30s limit)"
        result["return_code"] = -1
    except Exception as e:
        result["stderr"] = f"Error executing code: {str(e)}"
        result["return_code"] = -1

    return json.dumps(result, indent=2)


@tool
def create_tool_from_code(
    tool_name: str,
    tool_description: str,
    tool_code: str,
    save_to_file: bool = True,
    tools_dir: str = "outputs/tools"
) -> str:
    """
    Dynamically create a new tool from code.

    This enables the agent to extend its capabilities by creating new tools
    on the fly based on identified needs during research.

    Args:
        tool_name: Name for the new tool
        tool_description: Description of what the tool does
        tool_code: Python code implementing the tool function
        save_to_file: Whether to save the tool to a file
        tools_dir: Directory to save custom tools

    Returns:
        A JSON string with tool creation status and metadata
    """
    result = {
        "tool_name": tool_name,
        "description": tool_description,
        "status": "created",
        "file_path": None,
        "error": None
    }

    try:
        # Validate tool name
        if not tool_name.isidentifier():
            result["status"] = "error"
            result["error"] = f"Invalid tool name: {tool_name}"
            return json.dumps(result, indent=2)

        # Create tools directory if needed
        if save_to_file:
            os.makedirs(tools_dir, exist_ok=True)

            # Create tool file with proper decorator
            tool_file_content = f'''"""
Dynamically created tool: {tool_name}
{tool_description}
"""

from smolagents import tool

@tool
def {tool_name}(*args, **kwargs):
    """
    {tool_description}
    """
{chr(10).join("    " + line for line in tool_code.strip().split(chr(10)))}
'''

            tool_file_path = Path(tools_dir) / f"{tool_name}.py"
            with open(tool_file_path, 'w') as f:
                f.write(tool_file_content)

            result["file_path"] = str(tool_file_path)

        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


@tool
def install_package(package_name: str) -> str:
    """
    Install a Python package using pip.

    Useful for adding dependencies needed for experiments.

    Args:
        package_name: Name of the package to install (can include version specifier)

    Returns:
        A JSON string with installation status
    """
    result = {
        "package": package_name,
        "status": "installed",
        "message": ""
    }

    try:
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )

        if process.returncode == 0:
            result["status"] = "success"
            result["message"] = f"Successfully installed {package_name}"
        else:
            result["status"] = "error"
            result["message"] = process.stderr

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)

    return json.dumps(result, indent=2)


class ExperimentRunner:
    """
    Manages experiment execution and results.
    """

    def __init__(self, output_dir: str = "outputs/experiments"):
        self.output_dir = output_dir
        self.experiments_history = []
        os.makedirs(output_dir, exist_ok=True)

    def run(self, code: str, language: str = "python") -> Dict:
        """Run an experiment and track results."""
        result_json = run_experiment(code, language, True, self.output_dir)
        result = json.loads(result_json)
        self.experiments_history.append(result)
        return result

    def get_history(self):
        """Get experiment history."""
        return self.experiments_history


class ToolCreator:
    """
    Facilitates dynamic tool creation.
    """

    def __init__(self, tools_dir: str = "outputs/tools"):
        self.tools_dir = tools_dir
        self.created_tools = []
        os.makedirs(tools_dir, exist_ok=True)

    def create(self, name: str, description: str, code: str) -> Dict:
        """Create a new tool."""
        result_json = create_tool_from_code(name, description, code, True, self.tools_dir)
        result = json.loads(result_json)
        if result["status"] == "success":
            self.created_tools.append(result)
        return result

    def list_tools(self):
        """List all created tools."""
        return self.created_tools
