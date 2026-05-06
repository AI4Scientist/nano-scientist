"""Utility functions for the Autonomous Scientist agent."""

import asyncio
import json
import os
import re
import yaml
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


_MODEL: str
_IN_COST: float
_OUT_COST: float
_BASE_URL: str
_EST_AVG_PROMPT_TOKENS: int
_EST_AVG_OUTPUT_TOKENS: int
_TOOL_DEFAULT_TIMEOUT: int
_TOOL_MAX_TIMEOUT: int
_TOOL_STDOUT_LIMIT: int
_TOOL_STDERR_LIMIT: int


def init_env(env_path: str = None):
    """Load environment variables from the specified .env file.

    If no path given, defaults to .env in the project root.
    """
    if env_path is None:
        env_path = str(Path(__file__).resolve().parents[1] / ".env")
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")
    load_dotenv(path, override=True)
    global _MODEL, _IN_COST, _OUT_COST, _BASE_URL, _EST_AVG_PROMPT_TOKENS, _EST_AVG_OUTPUT_TOKENS
    global _TOOL_DEFAULT_TIMEOUT, _TOOL_MAX_TIMEOUT, _TOOL_STDOUT_LIMIT, _TOOL_STDERR_LIMIT
    _MODEL                 = os.environ.get("MODEL_NAME", "z-ai/glm-5")
    _IN_COST               = float(os.environ.get("INPUT_TOKEN_COST_PER_MILLION", "0.95"))
    _OUT_COST              = float(os.environ.get("OUTPUT_TOKEN_COST_PER_MILLION", "2.55"))
    _BASE_URL              = os.environ.get("INFERENCE_BASE_URL", "https://openrouter.ai/api/v1")
    _EST_AVG_PROMPT_TOKENS = int(os.environ.get("EST_AVG_PROMPT_TOKENS", "500"))
    _EST_AVG_OUTPUT_TOKENS = int(os.environ.get("EST_AVG_OUTPUT_TOKENS", "300"))
    _TOOL_DEFAULT_TIMEOUT  = int(os.environ.get("TOOL_DEFAULT_TIMEOUT", "60"))
    _TOOL_MAX_TIMEOUT      = int(os.environ.get("TOOL_MAX_TIMEOUT",     "300"))
    _TOOL_STDOUT_LIMIT     = int(os.environ.get("TOOL_STDOUT_LIMIT",    "4000"))
    _TOOL_STDERR_LIMIT     = int(os.environ.get("TOOL_STDERR_LIMIT",    "1000"))


# --- API Key Registry ---
# Maps environment variable names to what they unlock for the scientist.
# Keys marked [REQUIRED] must be set — the agent cannot function without them.
API_KEY_REGISTRY = {
    # --- REQUIRED ---
    "OPENROUTER_API_KEY": "[REQUIRED] Core LLM inference — every node calls the LLM through OpenRouter; without this key the agent cannot run at all",
    # --- SKILL-GATED ---
    "S2_API_KEY":         "Required by paper-navigator (Semantic Scholar search and citation traversal)",
    "GITHUB_TOKEN":       "Required by paper-navigator and experiment skills (GitHub code/repo search)",
    "HF_TOKEN":           "Required by experiment skills (Hugging Face model/dataset discovery)"
}


def detect_api_keys() -> dict[str, bool]:
    """Check which API keys are available in the environment.

    Returns a dict of {key_name: is_set} for all known keys.
    Must be called AFTER init_env().
    """
    return {key: bool(os.environ.get(key)) for key in API_KEY_REGISTRY}


def format_available_keys(keys: dict[str, bool]) -> str:
    """Format detected API keys as a readable summary for LLM prompts."""
    lines = []
    for key, is_set in keys.items():
        status = "available" if is_set else "NOT SET"
        desc = API_KEY_REGISTRY.get(key, "")
        lines.append(f"- {key}: {status} — {desc}")
    return "\n".join(lines)


# --- Token counting ---
_tiktoken_enc = None

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base encoding).

    Falls back to rough word-based estimate if tiktoken is not installed.
    """
    global _tiktoken_enc
    if _TIKTOKEN_AVAILABLE:
        if _tiktoken_enc is None:
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return len(_tiktoken_enc.encode(text))
    # Fallback: ~1.3 tokens per word (rough estimate)
    return int(len(text.split()) * 1.3)


def estimate_calls_remaining(
    budget_remaining: float,
    cost_log: list = None,
    avg_prompt_tokens: int = None,
    avg_output_tokens: int = None,
) -> int:
    """Estimate how many LLM calls remain given current budget.

    If cost_log is provided, derives the average cost-per-call from actual
    observed usage rather than the hardcoded defaults.
    """
    if avg_prompt_tokens is None:
        avg_prompt_tokens = _EST_AVG_PROMPT_TOKENS
    if avg_output_tokens is None:
        avg_output_tokens = _EST_AVG_OUTPUT_TOKENS
    if cost_log:
        real_calls = [e for e in cost_log if e.get("input_tokens", 0) > 0]
        if real_calls:
            avg_prompt_tokens = int(sum(e["input_tokens"] for e in real_calls) / len(real_calls))
            avg_output_tokens = int(sum(e["output_tokens"] for e in real_calls) / len(real_calls))
    cost_per_call = (
        avg_prompt_tokens * _IN_COST / 1_000_000
        + avg_output_tokens * _OUT_COST / 1_000_000
    )
    if cost_per_call <= 0:
        return 0
    return max(0, int(budget_remaining / cost_per_call))


# Persistent system prompt injected into every LLM call
_SYSTEM_PROMPT_TEMPLATE = (
    "You are an autonomous research assistant. "
    "Your final goal is to generate a high-quality technical report on the assigned topic. "
    "~{calls} LLM calls remaining. "
    "Be concise and prioritise information density. "
    "Every token costs money — avoid padding, repetition, or lengthy preambles. "
    "IMPORTANT: Respond with plain text only. Do NOT use tool calls, function calls, "
    "XML tags, or any structured invocation syntax. Write your answer directly as text."
)


# Tool definition exposed to the model for Bash execution
_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash or python command in the task working directory. Use for data collection, computation, file I/O, and API calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 300).",
                    "default": 300,
                },
            },
            "required": ["command"],
        },
    },
}


def _execute_tool_call(tool_name: str, arguments: dict, cwd: str) -> str:
    """Execute a model-requested tool call and return its output as a string."""
    import subprocess
    if tool_name == "bash":
        command = arguments.get("command", "")
        # Strip leading `sleep N` calls — they waste tool rounds and cause timeouts.
        command = re.sub(r"^\s*sleep\s+\d+\s*&&\s*", "", command)
        timeout = min(int(arguments.get("timeout", _TOOL_DEFAULT_TIMEOUT)), _TOOL_MAX_TIMEOUT)
        try:
            r = subprocess.run(
                command, shell=True, cwd=cwd,
                capture_output=True, text=True,
                errors="replace", timeout=timeout,
                env={**os.environ},
            )
            out = r.stdout[:_TOOL_STDOUT_LIMIT]
            err = r.stderr[:_TOOL_STDERR_LIMIT]
            result = f"exit={r.returncode}"
            if out:
                result += f"\n{out}"
            if err:
                result += f"\n[stderr] {err}"
            return result
        except subprocess.TimeoutExpired:
            return f"[ERROR] Command timed out after {timeout}s"
        except Exception as e:
            return f"[ERROR] {e}"
    return f"[ERROR] Unknown tool: {tool_name}"


def get_async_client() -> AsyncOpenAI:
    """Create async OpenAI-compatible client from env config."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found.\n"
            "Add to your .env file:\n"
            "  OPENROUTER_API_KEY=sk-or-v1-your-key-here\n"
            "Get one at https://openrouter.ai/keys"
        )
    return AsyncOpenAI(
        base_url=_BASE_URL,
        api_key=api_key,
    )


async def call_llm_async(
    prompt: str,
    system: str = None,
    budget_remaining: float = None,
    allow_tools: bool = False,
    cost_log: list = None,
) -> tuple[str, dict]:
    """Async version of call_llm."""
    calls_left = estimate_calls_remaining(budget_remaining or 0, cost_log=cost_log)
    budget_ctx = _SYSTEM_PROMPT_TEMPLATE.format(calls=calls_left)
    full_system = budget_ctx + ("\n\n" + system if system else "")
    est_input = count_tokens(full_system) + count_tokens(prompt)

    client = get_async_client()
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": prompt},
    ]
    kwargs = {"model": _MODEL, "messages": messages}
    if not allow_tools:
        kwargs["tool_choice"] = "none"

    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as e:
        if not allow_tools and ("tool_choice" in str(e).lower() or getattr(getattr(e, "response", None), "status_code", None) == 400):
            response = await client.chat.completions.create(model=_MODEL, messages=messages)
        else:
            raise

    if not response or not response.choices:
        raise RuntimeError(f"LLM returned empty response (no choices): {response!r}")
    choice = response.choices[0]
    text = choice.message.content or ""
    if not allow_tools:
        if not text and choice.message.tool_calls:
            text = "\n".join(
                tc.function.arguments for tc in choice.message.tool_calls
                if tc.function and tc.function.arguments
            )
        text = re.sub(r"<[a-zA-Z0-9_:]+:tool_call>.*?</[a-zA-Z0-9_:]+:tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?[a-zA-Z0-9_:]+:tool_call[^>]*>", "", text)
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else est_input
    output_tokens = usage.completion_tokens if usage else count_tokens(text)
    cost = (
        input_tokens * _IN_COST / 1_000_000
        + output_tokens * _OUT_COST / 1_000_000
    )
    return text, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "estimated_input_tokens": est_input,
    }


async def call_llm_with_tools_async(
    prompt: str,
    system: str = None,
    budget_remaining: float = None,
    cwd: str = ".",
    max_tool_rounds: int = None,
    cost_log: list = None,
) -> tuple[str, dict]:
    """Async version of call_llm_with_tools. Tool execution runs in a thread pool."""
    if max_tool_rounds is None:
        max_tool_rounds = int(os.environ.get("MAX_TOOL_ROUNDS", "16"))
    calls_left = estimate_calls_remaining(budget_remaining or 0, cost_log=cost_log)
    budget_ctx = _SYSTEM_PROMPT_TEMPLATE.format(calls=calls_left)
    full_system = budget_ctx + ("\n\n" + system if system else "")

    client = get_async_client()
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": prompt},
    ]

    total_input = count_tokens(full_system) + count_tokens(prompt)
    total_output = 0
    total_cost = 0.0
    final_text = ""

    for round_i in range(max_tool_rounds):
        try:
            response = await client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                tools=[_BASH_TOOL],
                tool_choice="auto",
            )
        except Exception as e:
            if "tool" in str(e).lower() or getattr(getattr(e, "response", None), "status_code", None) == 400:
                plain_text, plain_usage = await call_llm_async(prompt, system=system, budget_remaining=budget_remaining)
                return plain_text, plain_usage
            raise

        usage = response.usage
        if usage:
            total_input += usage.prompt_tokens
            total_output += usage.completion_tokens
            total_cost += (
                usage.prompt_tokens * _IN_COST / 1_000_000
                + usage.completion_tokens * _OUT_COST / 1_000_000
            )

        choice = response.choices[0]
        msg = choice.message
        if msg.content:
            final_text = msg.content
        if not msg.tool_calls:
            break

        messages.append(msg)
        tool_results = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            # Run blocking tool execution in thread pool to not block event loop
            output = await asyncio.get_event_loop().run_in_executor(
                None, _execute_tool_call, tc.function.name, args, cwd
            )
            print(f"[tool:{tc.function.name}] {str(args.get('command',''))[:60]} → {output[:80]}")
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })
        messages.extend(tool_results)

        if round_i == max_tool_rounds - 1:
            print(f"[call_llm_with_tools_async] Reached max_tool_rounds={max_tool_rounds}, stopping.")
            def _msg_role(m):
                return m["role"] if isinstance(m, dict) else getattr(m, "role", None)
            def _msg_content(m):
                return m["content"] if isinstance(m, dict) else getattr(m, "content", None)
            tool_outputs = "\n\n".join(
                _msg_content(m) for m in messages
                if _msg_role(m) == "tool" and _msg_content(m)
            )
            return final_text, {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost": total_cost,
                "estimated_input_tokens": total_input,
                "tool_rounds_exhausted": True,
                "tool_outputs": tool_outputs,
            }

    return final_text, {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost": total_cost,
        "estimated_input_tokens": total_input,
        "tool_rounds_exhausted": False,
    }


def load_skill_index(skills_dir: str) -> dict[str, str]:
    """Load skill index from skills.json — lightweight {name: description}.

    skills.json acts as a DB index: only descriptions are loaded,
    full SKILL.md content is loaded on demand via load_skill_content().
    """
    index_path = Path(skills_dir) / "skills.json"
    if not index_path.exists():
        raise ValueError(f"skills.json not found at {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    index = {}
    for skill in data.get("skills", []):
        index[skill["id"]] = skill.get("description", skill["id"])
    if not index:
        raise ValueError(f"No skills found in {index_path}")
    return index


def _parse_skill_frontmatter(raw: str) -> dict:
    """Parse YAML frontmatter from a SKILL.md string. Returns {} on failure."""
    fm = re.match(r'^---\s*\n(.*?)\n---\s*\n', raw, re.DOTALL)
    if not fm:
        return {}
    try:
        meta = yaml.safe_load(fm.group(1)) or {}
    except yaml.YAMLError:
        return {}
    for key in ("required-keys", "allowed-tools"):
        val = meta.get(key, [])
        if isinstance(val, str):
            val = [v.strip() for v in val.split(",") if v.strip()]
        meta[key] = val
    return meta


def filter_skill_index(skills_dir: str, index: dict[str, str], api_keys: dict[str, bool]) -> dict[str, str]:
    """Remove skills whose required-keys are not available.

    Reads each SKILL.md frontmatter (only the first few lines) to check
    required-keys. Skills with no required-keys are always included.
    """
    filtered = {}
    for skill_id, desc in index.items():
        skill_file = Path(skills_dir) / skill_id / "SKILL.md"
        meta = _parse_skill_frontmatter(skill_file.read_text(encoding="utf-8")) if skill_file.exists() else {}
        required = meta.get("required-keys", [])
        if all(api_keys.get(k) for k in required):
            filtered[skill_id] = desc
    return filtered


def load_skill_content(skills_dir: str, skill_name: str) -> tuple[str, dict]:
    """Lazy-load a single SKILL.md on demand (called by ExecuteSkill only).

    Returns (content, metadata) where metadata is the parsed YAML frontmatter.
    Content has the frontmatter stripped.
    """
    skill_file = Path(skills_dir) / skill_name / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"SKILL.md not found: {skill_file}")
    raw = skill_file.read_text(encoding="utf-8")
    metadata = _parse_skill_frontmatter(raw)
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', raw, re.DOTALL)
    content = raw[fm_match.end():] if fm_match else raw
    return content.strip(), metadata


def format_skill_index(index: dict[str, str]) -> str:
    """Format the skill index as a readable list for LLM prompts."""
    lines = []
    for name, desc in sorted(index.items()):
        short = desc[:117] + "..." if len(desc) > 120 else desc
        lines.append(f"- {name}: {short}")
    return "\n".join(lines)


def parse_yaml_response(text: str):
    """Extract and parse a YAML block from an LLM response.

    Returns a dict, list, or None. Callers must check the type they expect.
    """
    # Try ```yaml, ```yml, ```YAML
    match = re.search(r"```(?:yaml|yml)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        # Try any fenced block that looks like YAML (dict key: or list - item)
        for m in re.finditer(r"```\w*\s*(.*?)```", text, re.DOTALL):
            block_candidate = m.group(1)
            if re.search(r"^\s*\w+\s*:", block_candidate, re.MULTILINE) or \
               re.search(r"^\s*-\s+\w+", block_candidate, re.MULTILINE):
                match = m
                break
    block = match.group(1).strip() if match else text.strip()
    try:
        result = yaml.safe_load(block)
        if isinstance(result, (dict, list)):
            return result
    except yaml.YAMLError:
        pass

    # Last resort: find first line starting with "- " and parse from there
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("- "):
            candidate = "\n".join(lines[i:])
            try:
                result = yaml.safe_load(candidate)
                if isinstance(result, (dict, list)):
                    return result
            except yaml.YAMLError:
                break
    return None


def extract_bibtex(text: str) -> tuple[str, list[str]]:
    """Split LLM response into (main_content, list_of_bibtex_entries).

    Handles multiple LLM output formats:
    - ```bibtex ... ```  (standard)
    - ```bib ... ```     (common variant)
    - ``` ... ```        (no language tag, if it contains @article etc.)
    - Raw @article{...}  entries outside code blocks (fallback)
    """
    # Try fenced code blocks: ```bibtex, ```bib, ```BibTeX, ```latex (with bibtex inside)
    bib_match = re.search(r"```(?:bibtex|bib|BibTeX)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)

    if not bib_match:
        # Try generic fenced block that contains @article/@inproceedings etc.
        for m in re.finditer(r"```\w*\s*(.*?)```", text, re.DOTALL):
            if re.search(r"@\w+\{", m.group(1)):
                bib_match = m
                break

    if bib_match:
        main_content = text[: bib_match.start()].strip()
        bib_block = bib_match.group(1).strip()
    else:
        # Fallback: extract raw @type{...} entries from the end of the text
        raw_entries = list(re.finditer(r"@\w+\{", text))
        if raw_entries:
            first_entry_pos = raw_entries[0].start()
            main_content = text[:first_entry_pos].strip()
            bib_block = text[first_entry_pos:].strip()
        else:
            return text.strip(), []

    # Split into individual entries
    entries = re.findall(r"(@\w+\{[^@]+)", bib_block, re.DOTALL)
    entries = [e.strip() for e in entries if e.strip()]
    return main_content, entries


def is_valid_bibtex_key(key: str) -> bool:
    """Validate a BibTeX citation key.

    Valid keys: alphanumeric plus hyphen, underscore, colon, period, slash.
    Must be 2-80 chars. No spaces, no quotes, no prose fragments.
    Rejects internal filename-style keys like dataset_metadata, classification_results.
    """
    if not key or len(key) < 2 or len(key) > 80:
        return False
    if not re.match(r'^[a-zA-Z0-9_:.\-/]+$', key):
        return False
    # Reject keys with too many consecutive lowercase letters (likely prose)
    if len(key) > 40 and not re.search(r'\d', key):
        return False
    # Reject filename-style keys: all lowercase with underscores and no digits
    # e.g. dataset_metadata, classification_results, structured_metadata_complete
    # Real citation keys follow author+year or abbreviated title patterns
    if re.match(r'^[a-z][a-z_]+$', key) and '_' in key and not re.search(r'\d', key):
        # Allow short snake_case that could be a legitimate abbreviation (<=20 chars)
        # Reject long descriptive snake_case filenames (>20 chars)
        if len(key) > 20:
            return False
    return True


def _bibtex_entry_has_required_fields(entry: str) -> bool:
    """Check that a BibTeX entry has at minimum author and title fields."""
    lower = entry.lower()
    has_author = bool(re.search(r'author\s*=', lower))
    has_title = bool(re.search(r'title\s*=', lower))
    return has_author and has_title


def dedup_bibtex(entries: list[str]) -> str:
    """Deduplicate BibTeX entries by cite key, return combined .bib content.

    Rejects entries with invalid cite keys or missing required fields.
    """
    seen = {}
    rejected = 0
    for entry in entries:
        match = re.match(r"@\w+\{([^,]+),", entry)
        if match:
            key = match.group(1).strip()
            if not is_valid_bibtex_key(key):
                rejected += 1
                continue
            if not _bibtex_entry_has_required_fields(entry):
                rejected += 1
                continue
            if key not in seen:
                seen[key] = entry
    if rejected:
        print(f"[dedup_bibtex] Rejected {rejected} invalid BibTeX entries")
    return "\n\n".join(seen.values()) + "\n" if seen else ""


def track_cost(shared: dict, step: str, usage: dict):
    """Append cost to shared ledger and decrement remaining budget."""
    shared.setdefault("cost_log", []).append({"step": step, **usage})
    shared["budget_remaining"] = shared.get("budget_remaining", 0) - usage["cost"]
