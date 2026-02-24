"""PocketFlow nodes for the Autonomous Scientist agent."""

import os
import re
import subprocess
import uuid
from collections import Counter
from pathlib import Path

from pocketflow import Node

from .utils import (
    call_llm,
    format_skill_index,
    format_available_keys,
    load_skill_content,
    load_quality_standard,
    parse_yaml_response,
    extract_bibtex,
    dedup_bibtex,
    track_cost,
)

# ---------------------------------------------------------------------------
# Budget reserves — enough for WriteTeX + CompileTeX + one FixTeX round
# ---------------------------------------------------------------------------
BUDGET_RESERVE = 0.03  # dollars


# ---------------------------------------------------------------------------
# LaTeX skeleton — hardcoded, known-compilable with pdflatex + natbib
# ---------------------------------------------------------------------------
LATEX_SKELETON = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\graphicspath{{./figures/}}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xcolor}

\title{%% TITLE %%}
\author{Autonomous Scientist Agent}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
%% ABSTRACT %%
\end{abstract}

%% BODY %%

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
"""


# ===================================================================
# 1. BudgetPlanner
# ===================================================================
class BudgetPlanner(Node):
    """Analyze topic + budget → produce a prioritized research plan."""

    def prep(self, shared):
        return {
            "topic": shared["topic"],
            "budget": shared["budget_dollars"],
            "skills": format_skill_index(shared["skill_index"]),
            "quality_standard": shared.get("quality_standard", ""),
            "api_keys": format_available_keys(shared.get("api_keys", {})),
        }

    def exec(self, prep_res):
        # Extract adaptive structure section from quality standard if available
        quality_guidance = ""
        qs = prep_res["quality_standard"]
        if qs:
            # Pull Section 2.3 (adaptive structure) and Section 4 (citation quality)
            import re as _re
            adaptive = _re.search(
                r"### 2\.3 Adaptive Structure by Report Type.*?(?=\n## |\n### [^2]|\Z)",
                qs, _re.DOTALL,
            )
            citation = _re.search(
                r"## 4\. Citation Quality.*?(?=\n## [^4]|\Z)",
                qs, _re.DOTALL,
            )
            parts = []
            if adaptive:
                parts.append(adaptive.group(0).strip())
            if citation:
                parts.append(citation.group(0).strip())
            if parts:
                quality_guidance = "\n\n## Paper Quality Standard (excerpt)\n" + "\n\n".join(parts)

        prompt = f"""You are a research planning assistant. Given a research topic and a dollar budget for LLM inference, produce an ordered plan of research skills to execute.

## Research Topic
{prep_res["topic"]}

## Budget
${prep_res["budget"]:.2f} USD

## Cost Model
Each skill execution costs roughly $0.003-$0.005.
Each planning/decision step costs roughly $0.001.
The final LaTeX report generation costs roughly $0.01.
Always reserve $0.03 for the final report + compilation.
IMPORTANT: Plan MANY steps to use the budget effectively. A $1 budget supports ~200 skill calls. A $20 budget supports ~4000 skill calls.

## Available API Keys
{prep_res["api_keys"]}

Only plan skills whose required API keys are available. Do not plan skills that depend on missing keys.

## Available Skills
{prep_res["skills"]}

## Budget Strategy Guidelines
- Budget < $0.10: research-lookup (1 call), then write report (Quick Summary).
- Budget $0.10-$0.50: research-lookup (2-3 calls on subtopics) + literature-review, then write report (Literature Review).
- Budget $0.50-$2.00: research-lookup (3-5 calls) + literature-review + hypothesis-generation + scientific-critical-thinking (Research Report).
- Budget $2.00-$5.00: Multiple research-lookup calls (5-10, each on different subtopics) + literature-review + hypothesis-generation + scientific-critical-thinking + statistical-analysis + scholar-evaluation + peer-review (Full Paper).
- Budget $5.00+: All of the above PLUS repeated research-lookup on each research question, multiple literature-review passes on subtopics, data-visualization, scientific-slides, and venue-templates. Plan 20+ skill executions minimum. Use the budget to build deep, comprehensive research.

## Key Planning Rules
1. For budgets >= $2.00, plan AT LEAST 15 skill steps.
2. Use research-lookup MULTIPLE TIMES with different queries to gather comprehensive material.
3. Use literature-review on specific subtopics, not just the broad topic.
4. Every skill execution builds material and citations for the final report — more executions = better paper.
5. The decision engine can extend the plan beyond what you specify here, so focus on the most important initial steps.
{quality_guidance}

## Instructions
Produce a YAML plan. Each step has: step number, skill name, and a short reason.
Only include skills that fit within the budget after reserving $0.03 for the report.
Plan should produce enough material for the report type matching this budget tier.

```yaml
domain: <one-line topic classification>
report_type: <Quick Summary | Literature Review | Research Report | Full Paper>
plan:
  - step: 1
    skill: <skill-name>
    reason: <why this step>
  - step: 2
    skill: <skill-name>
    reason: <why this step>
```"""
        text, usage = call_llm(prompt)
        return text, usage

    def post(self, shared, prep_res, exec_res):
        text, usage = exec_res
        track_cost(shared, "budget_planner", usage)

        parsed = parse_yaml_response(text)
        shared["plan"] = parsed.get("plan", [])
        shared["domain"] = parsed.get("domain", "general")
        shared["report_type"] = parsed.get("report_type", "Literature Review")
        shared["budget_remaining"] = shared["budget_dollars"] - usage["cost"]
        shared["artifacts"] = {}
        shared["bibtex_entries"] = []
        shared["history"] = []
        shared["fix_attempts"] = 0

        # Create task directory early so all phases can persist intermediaries
        task_id = str(uuid.uuid4())
        out_dir = Path(shared.get("output_dir", "outputs")) / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "artifacts").mkdir(exist_ok=True)
        (out_dir / "figures").mkdir(exist_ok=True)
        (out_dir / "data").mkdir(exist_ok=True)
        (out_dir / "scripts").mkdir(exist_ok=True)
        shared["output_path"] = str(out_dir)

        # Persist plan
        import yaml as _yaml
        plan_data = {
            "task_id": task_id,
            "topic": shared["topic"],
            "domain": shared["domain"],
            "report_type": shared["report_type"],
            "budget_dollars": shared["budget_dollars"],
            "plan": shared["plan"],
        }
        (out_dir / "plan.yaml").write_text(
            _yaml.dump(plan_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

        print(f"[BudgetPlanner] Task directory: {out_dir}")
        print(f"[BudgetPlanner] Domain: {shared['domain']}")
        print(f"[BudgetPlanner] Report type: {shared['report_type']}")
        print(f"[BudgetPlanner] Plan: {len(shared['plan'])} steps")
        for s in shared["plan"]:
            print(f"  {s['step']}. {s['skill']} — {s.get('reason', '')}")
        print(f"[BudgetPlanner] Budget remaining: ${shared['budget_remaining']:.4f}")
        return "execute"


# ===================================================================
# 2. DecideNext — the agent core
# ===================================================================
class DecideNext(Node):
    """Pick the next skill to execute or decide to write the report."""

    def prep(self, shared):
        # Build compact history summary
        history_lines = []
        for h in shared.get("history", []):
            history_lines.append(f"- {h['skill']}: {h['summary']} (${h['cost']:.4f})")
        history_text = "\n".join(history_lines) if history_lines else "None yet."

        # Remaining plan steps — count-based to preserve duplicate skills
        exec_counts = Counter(h["skill"] for h in shared.get("history", []))
        remaining = []
        skill_seen = Counter()
        for s in shared.get("plan", []):
            skill = s["skill"]
            skill_seen[skill] += 1
            if skill_seen[skill] > exec_counts.get(skill, 0):
                remaining.append(s)

        return {
            "topic": shared["topic"],
            "remaining_plan": remaining,
            "history": history_text,
            "budget_remaining": shared.get("budget_remaining", 0),
            "artifact_keys": list(shared.get("artifacts", {}).keys()),
            "available_skills": format_skill_index(shared["skill_index"]),
        }

    def exec(self, prep_res):
        # Force write_tex if budget is too low
        if prep_res["budget_remaining"] < BUDGET_RESERVE:
            return {"action": "write_tex", "reason": "budget exhausted"}, {"input_tokens": 0, "output_tokens": 0, "cost": 0}

        remaining_yaml = "\n".join(
            f"  - {s['skill']}: {s.get('reason', '')}"
            for s in prep_res["remaining_plan"]
        ) if prep_res["remaining_plan"] else "All planned steps completed."

        # Calculate how many more skill calls the budget can support
        cost_per_skill = 0.005  # conservative estimate
        usable_budget = prep_res["budget_remaining"] - BUDGET_RESERVE
        affordable_steps = max(0, int(usable_budget / cost_per_skill))

        prompt = f"""You are the decision engine of an autonomous research agent.

## Research Topic
{prep_res["topic"]}

## Completed Steps
{prep_res["history"]}

## Remaining Planned Steps
{remaining_yaml}

## Budget Remaining
${prep_res["budget_remaining"]:.4f} (reserve $0.03 for final report)
Estimated affordable additional skill calls: {affordable_steps}

## Artifacts Collected
{', '.join(prep_res["artifact_keys"]) if prep_res["artifact_keys"] else 'None yet.'}

## Available Skills (for extending the plan)
{prep_res["available_skills"]}

## Instructions
Decide the next action. You may:
1. Execute a planned skill ("execute_skill") — pick from remaining plan
2. Execute an ADDITIONAL skill ("execute_skill") — if the plan is done but budget allows deeper research, propose a skill to strengthen the paper (e.g., repeat research-lookup with different angles, add peer-review, add scientific-critical-thinking, deepen literature-review on subtopics)
3. Write the final report ("write_tex") — ONLY when you have enough material AND less than 5 affordable steps remain

**Important**: Do NOT write the report early if there is substantial budget remaining. Use the budget to deepen research, gather more citations, and improve paper quality. A Full Paper needs 20+ citations and coverage of all mandatory sections.

Return YAML:
```yaml
action: execute_skill OR write_tex
skill: <skill-name>
reason: <brief reason>
```"""
        text, usage = call_llm(prompt)
        parsed = parse_yaml_response(text)
        return parsed, usage

    def post(self, shared, prep_res, exec_res):
        decision, usage = exec_res
        track_cost(shared, "decide_next", usage)

        if not decision or not isinstance(decision, dict):
            print("[DecideNext] WARNING: Failed to parse LLM response, defaulting to next plan step")
            remaining = prep_res.get("remaining_plan", [])
            if remaining:
                decision = {"action": "execute_skill", "skill": remaining[0]["skill"], "reason": "parse fallback"}
            else:
                decision = {"action": "write_tex", "reason": "parse fallback — no remaining steps"}

        action = decision.get("action", "write_tex")
        reason = decision.get("reason", "")
        print(f"[DecideNext] Action: {action} — {reason}")
        print(f"[DecideNext] Budget remaining: ${shared['budget_remaining']:.4f}")

        # Persist decision log to task directory
        shared.setdefault("decisions", []).append({
            "action": action,
            "skill": decision.get("skill", ""),
            "reason": reason,
            "budget_remaining": shared["budget_remaining"],
        })
        import json as _json
        out_dir = Path(shared["output_path"])
        (out_dir / "decisions.json").write_text(
            _json.dumps(shared["decisions"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Budget guard
        if shared["budget_remaining"] < BUDGET_RESERVE:
            print("[DecideNext] Budget guard triggered → write_tex")
            return "write_tex"

        if action == "execute_skill":
            shared["next_skill"] = decision.get("skill", "")
            return "execute_skill"
        return "write_tex"


# ===================================================================
# 3. ExecuteSkill
# ===================================================================
class ExecuteSkill(Node):
    """Load a skill's SKILL.md, run it via LLM, and execute any code blocks."""

    def prep(self, shared):
        skill_name = shared["next_skill"]
        # Lazy-load: read the SKILL.md and parse metadata
        skill_content, skill_metadata = load_skill_content(shared["skills_dir"], skill_name)

        # Detect code execution capability
        allowed_tools = skill_metadata.get("allowed-tools", [])
        can_execute = "Bash" in allowed_tools

        # Find available scripts for this skill
        scripts_dir = Path(shared["skills_dir"]) / skill_name / "scripts"
        available_scripts = []
        if scripts_dir.is_dir():
            available_scripts = [f.name for f in scripts_dir.iterdir()
                                 if f.suffix == ".py" and f.is_file()]

        # Condensed prior context (summaries only, not full artifacts)
        context_lines = []
        for h in shared.get("history", []):
            context_lines.append(f"### {h['skill']}\n{h['summary']}")
        prior_context = "\n\n".join(context_lines) if context_lines else "No prior research yet."

        # Collect existing generated data files for context
        generated_files = shared.get("generated_files", {})
        data_files_info = ""
        if generated_files:
            file_lines = []
            for sk, files in generated_files.items():
                for f in files:
                    file_lines.append(f"  - {f} (from {sk})")
            if file_lines:
                data_files_info = "Previously generated data files:\n" + "\n".join(file_lines)

        return {
            "skill_name": skill_name,
            "skill_content": skill_content,
            "topic": shared["topic"],
            "prior_context": prior_context,
            "can_execute": can_execute,
            "available_scripts": available_scripts,
            "scripts_dir": str(scripts_dir) if scripts_dir.is_dir() else "",
            "task_dir": shared.get("output_path", ""),
            "data_files_info": data_files_info,
        }

    def exec(self, prep_res):
        # Build code execution instructions if the skill supports it
        code_exec_block = ""
        if prep_res["can_execute"] and prep_res["task_dir"]:
            scripts_info = ""
            if prep_res["available_scripts"]:
                scripts_info = f"""
### Available Skill Scripts (in {prep_res['scripts_dir']})
These scripts are ready to use. Call them with `python {prep_res['scripts_dir']}/<script_name>`:
{chr(10).join(f'- {s}' for s in prep_res['available_scripts'])}
"""
            data_info = ""
            if prep_res["data_files_info"]:
                data_info = f"""
### Previously Generated Data
{prep_res['data_files_info']}
You can read these files in your code for further analysis or visualization.
"""

            code_exec_block = f"""

## Code Execution Available
You can include executable code to collect REAL data, generate REAL figures, or run REAL analyses.
Your working directory is: {prep_res['task_dir']}
{scripts_info}{data_info}
### How to include executable code
Place code between these markers. Supported: python, bash.

%%BEGIN CODE:python%%
# Your Python code here
# Save data to: {prep_res['task_dir']}/data/
# Save figures to: {prep_res['task_dir']}/figures/
%%END CODE%%

%%BEGIN CODE:bash%%
# Your bash commands here
%%END CODE%%

### Code Guidelines
- Save data files (CSV, JSON) to `{prep_res['task_dir']}/data/`
- Save figure files (PNG, PDF) to `{prep_res['task_dir']}/figures/`
- For figures: use matplotlib with `plt.savefig()` — do NOT use `plt.show()`
- Use descriptive filenames relating to the research topic
- Available libraries: matplotlib, pandas, numpy, seaborn, requests, scipy
- API tokens available as env vars: GITHUB_TOKEN, OPENROUTER_API_KEY, PERPLEXITY_API_KEY
- Timeout: 300 seconds — keep code focused and efficient
- Print a summary of collected/generated data to stdout
- IMPORTANT: You MUST include code blocks to produce real data and figures. Do NOT just describe what code would do — actually write it so it runs.
"""

        prompt = f"""You are executing a research skill as part of an autonomous scientist agent.

## Research Topic
{prep_res["topic"]}

## Prior Research Context
{prep_res["prior_context"]}

## Skill Instructions
Follow these instructions to produce your deliverable:

---
{prep_res["skill_content"]}
---
{code_exec_block}
## Citation Quality Requirements
- Include references to real, well-known papers in the field.
- Aim for breadth: cite multiple research groups, not just one lab.
- Include foundational/seminal papers and recent work (last 5 years when possible).
- Every claim or finding you mention should be backed by a citation.
- Target at least 5-10 references per skill execution.

## Output Format
1. Produce the skill's deliverable as detailed text.
2. At the VERY END of your response, you MUST include a BibTeX section between markers:

%%BEGIN BIBTEX%%
@article{{authorYYYYkeyword,
  author = {{Last, First and Last, First}},
  title = {{Full Paper Title}},
  journal = {{Journal Name}},
  year = {{YYYY}},
  volume = {{N}},
  pages = {{1--10}}
}}
%%END BIBTEX%%

3. Each BibTeX entry MUST have: author, title, year, and venue (journal or booktitle).
4. Use realistic cite keys: author2024keyword (e.g., smith2023attention).
5. Include one entry for EVERY paper you reference in your text.
6. This section is MANDATORY — do not skip it.

Begin your work now."""
        text, usage = call_llm(prompt)
        return text, usage

    def post(self, shared, prep_res, exec_res):
        text, usage = exec_res
        skill_name = prep_res["skill_name"]
        track_cost(shared, f"execute_skill:{skill_name}", usage)

        # --- Extract and execute code blocks ---
        code_outputs = []

        if prep_res["can_execute"] and prep_res["task_dir"]:
            task_dir = Path(prep_res["task_dir"])

            # Ensure subdirs exist
            (task_dir / "data").mkdir(exist_ok=True)
            (task_dir / "figures").mkdir(exist_ok=True)
            (task_dir / "scripts").mkdir(exist_ok=True)

            # Extract code blocks: %%BEGIN CODE:lang%% ... %%END CODE%%
            code_blocks = re.findall(
                r"%%BEGIN CODE:(\w+)%%(.*?)%%END CODE%%", text, re.DOTALL
            )

            for i, (lang, code) in enumerate(code_blocks):
                code = code.strip()
                if not code:
                    continue

                # Write script to task_dir/scripts/
                step_num = len(shared.get("history", [])) + 1
                ext = ".py" if lang == "python" else ".sh"
                script_path = task_dir / "scripts" / f"{step_num:02d}_{skill_name}_{i:02d}{ext}"
                script_path.write_text(code, encoding="utf-8")

                # Execute
                cmd = ["python", str(script_path)] if lang == "python" else ["bash", str(script_path)]
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=str(task_dir),
                        capture_output=True,
                        text=True,
                        errors="replace",
                        timeout=300,
                        env={**os.environ},
                    )
                    stdout = result.stdout[:3000]
                    stderr = result.stderr[:1000]
                    code_outputs.append(f"[Script {script_path.name}] exit={result.returncode}\n{stdout}")
                    if result.returncode != 0:
                        code_outputs.append(f"[STDERR] {stderr}")
                    print(f"[ExecuteSkill] Ran {script_path.name}: exit={result.returncode}")
                except subprocess.TimeoutExpired:
                    code_outputs.append(f"[Script {script_path.name}] TIMEOUT after 300s")
                    print(f"[ExecuteSkill] Script {script_path.name} timed out")
                except Exception as e:
                    code_outputs.append(f"[Script {script_path.name}] ERROR: {e}")
                    print(f"[ExecuteSkill] Script {script_path.name} failed: {e}")

            # Scan for generated files
            generated_files = []
            for subdir in ["data", "figures"]:
                scan_dir = task_dir / subdir
                if scan_dir.is_dir():
                    for f in sorted(scan_dir.iterdir()):
                        if f.is_file():
                            generated_files.append(str(f))
            if generated_files:
                shared.setdefault("generated_files", {})[skill_name] = generated_files
                for gf in generated_files:
                    print(f"[ExecuteSkill] Generated: {gf}")

        # --- Extract BibTeX ---
        bibtex_match = re.search(r"%%BEGIN BIBTEX%%(.*?)%%END BIBTEX%%", text, re.DOTALL)
        if bibtex_match:
            main_content = text[:bibtex_match.start()].strip()
            bib_block = bibtex_match.group(1).strip()
            bib_entries = re.findall(r"(@\w+\{[^@]+)", bib_block, re.DOTALL)
            bib_entries = [e.strip() for e in bib_entries if e.strip()]
        else:
            # Fallback: try fenced code blocks and raw entries
            main_content, bib_entries = extract_bibtex(text)

        # Remove code blocks from main content for cleaner artifact storage
        main_content = re.sub(
            r"%%BEGIN CODE:\w+%%.*?%%END CODE%%", "", main_content, flags=re.DOTALL
        ).strip()

        # Append code execution results to the artifact
        if code_outputs:
            main_content += "\n\n## Code Execution Results\n" + "\n".join(code_outputs)

        shared["artifacts"][skill_name] = main_content
        shared["bibtex_entries"].extend(bib_entries)

        # Create short summary for history (first 300 chars)
        summary = main_content[:300].replace("\n", " ")
        if len(main_content) > 300:
            summary += "..."
        step_num = len(shared["history"]) + 1
        shared["history"].append({
            "step": step_num,
            "skill": skill_name,
            "summary": summary,
            "cost": usage["cost"],
        })

        # Persist full artifact and BibTeX to task directory
        out_dir = Path(shared["output_path"])
        artifact_dir = out_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        artifact_file = artifact_dir / f"{step_num:02d}_{skill_name}.md"
        artifact_file.write_text(main_content, encoding="utf-8")
        if bib_entries:
            bib_file = artifact_dir / f"{step_num:02d}_{skill_name}.bib"
            bib_file.write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")

        # Persist accumulated history snapshot
        import json as _json
        (out_dir / "history.json").write_text(
            _json.dumps(shared["history"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[ExecuteSkill] Completed: {skill_name} (${usage['cost']:.4f})")
        print(f"[ExecuteSkill] Saved: {artifact_file.name}")
        print(f"[ExecuteSkill] BibTeX entries found: {len(bib_entries)}")
        print(f"[ExecuteSkill] Budget remaining: ${shared['budget_remaining']:.4f}")
        return "decide"


# ===================================================================
# 4. WriteTeX
# ===================================================================
class WriteTeX(Node):
    """Synthesize all artifacts into compilable .tex + .bib files."""

    def prep(self, shared):
        # Collect all cite keys for the LLM to reference
        cite_keys = []
        for entry in shared.get("bibtex_entries", []):
            m = re.match(r"@\w+\{([^,]+),", entry)
            if m:
                cite_keys.append(m.group(1).strip())

        # Determine which sections have content based on report type
        report_type = shared.get("report_type", "Literature Review")
        has_methods = any(
            k in shared.get("artifacts", {})
            for k in ("statistical-analysis", "method-implementation", "experimental-evaluation")
        )
        has_results = has_methods  # results typically accompany methods

        # Extract writing guidelines from quality standard
        writing_guide = ""
        qs = shared.get("quality_standard", "")
        if qs:
            # Pull sections 2.1, 3, and 6 (structure, writing rules, checklist)
            section_req = re.search(
                r"### 2\.1 Mandatory Sections.*?(?=\n### 2\.2 |\Z)",
                qs, re.DOTALL,
            )
            writing_rules = re.search(
                r"## 3\. Writing Quality Rules.*?(?=\n## 4\.|\Z)",
                qs, re.DOTALL,
            )
            checklist = re.search(
                r"## 6\. Self-Assessment Checklist.*?(?=\n## Sources|\n---|\Z)",
                qs, re.DOTALL,
            )
            parts = []
            if section_req:
                parts.append(section_req.group(0).strip())
            if writing_rules:
                parts.append(writing_rules.group(0).strip())
            if checklist:
                parts.append(checklist.group(0).strip())
            if parts:
                writing_guide = "\n\n".join(parts)

        # Scan for generated figures
        figure_files = []
        out_dir = Path(shared.get("output_path", ""))
        figures_dir = out_dir / "figures"
        if figures_dir.is_dir():
            for f in sorted(figures_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg"):
                    figure_files.append(f.name)

        # Scan for generated data files (for methods/results context)
        data_files = []
        data_dir = out_dir / "data"
        if data_dir.is_dir():
            for f in sorted(data_dir.iterdir()):
                if f.is_file():
                    data_files.append(f.name)

        return {
            "topic": shared["topic"],
            "artifacts": shared.get("artifacts", {}),
            "cite_keys": cite_keys,
            "has_methods": has_methods or bool(data_files),
            "has_results": has_results or bool(data_files),
            "report_type": report_type,
            "writing_guide": writing_guide,
            "figure_files": figure_files,
            "data_files": data_files,
        }

    def exec(self, prep_res):
        # Build context from artifacts
        artifact_text = ""
        for name, content in prep_res["artifacts"].items():
            artifact_text += f"\n\n### Artifact: {name}\n{content}"

        # Determine sections based on report type
        report_type = prep_res["report_type"]
        sections = ["abstract", "introduction", "background"]
        if report_type in ("Research Report", "Full Paper") and prep_res["has_methods"]:
            sections.append("methods")
        if report_type in ("Research Report", "Full Paper") and prep_res["has_results"]:
            sections.append("results")
        sections.extend(["discussion", "conclusion"])
        if report_type == "Full Paper":
            sections.append("limitations")

        cite_list = ", ".join(prep_res["cite_keys"]) if prep_res["cite_keys"] else "No citations available."

        # Build quality guidance block
        quality_block = ""
        if prep_res["writing_guide"]:
            quality_block = f"""
## Paper Quality Standard
You MUST follow these quality standards when writing. This is non-negotiable.

{prep_res["writing_guide"]}
"""

        # Build figure inclusion block
        figure_block = ""
        if prep_res.get("figure_files"):
            figure_list = "\n".join(f"- {f}" for f in prep_res["figure_files"])
            figure_block = f"""
## Available Figures
The following figures have been generated during research and are available for inclusion.
You MUST include them in the paper where they support the narrative.
{figure_list}

To include a figure, use this LaTeX pattern:
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{{figures/<filename>}}
\\caption{{Your descriptive caption here.}}
\\label{{fig:<short-label>}}
\\end{{figure}}

Reference each figure in the text as Figure~\\ref{{fig:<label>}}.
"""

        # Build data files context
        data_block = ""
        if prep_res.get("data_files"):
            data_list = "\n".join(f"- {f}" for f in prep_res["data_files"])
            data_block = f"""
## Collected Data Files
The following data files were collected during research. Reference their contents in your Methods and Results sections:
{data_list}
"""

        prompt = f"""You are writing a scientific {report_type.lower()} as compilable LaTeX.
{quality_block}
## Research Topic
{prep_res["topic"]}

## Research Artifacts (your source material)
{artifact_text}

## Available BibTeX cite keys
{cite_list}
{figure_block}{data_block}
## Report Type: {report_type}
## Sections to Write: {', '.join(sections)}

## STRUCTURAL RULES
1. **Title**: Specific, descriptive, under 15 words. Captures the central contribution.
2. **Abstract**: 150-250 words, self-contained, follows Context-Content-Conclusion structure. States problem, approach, findings, significance.
3. **Introduction**: Progress from broad context to specific gap. End with clear contribution statement.
4. **Background/Related Work**: Synthesize prior work by theme, not just list papers. Identify what is missing.
5. **Discussion**: Interpret results, compare with prior work, acknowledge limitations honestly.
6. **Conclusion**: 1-2 paragraphs max. State how work advances the field. Do NOT repeat the abstract.

## WRITING RULES
- Every paragraph follows Context-Content-Conclusion: first sentence sets context, body presents content, last sentence gives takeaway.
- Active voice preferred: "We propose X" not "X is proposed."
- No unsupported claims — every assertion needs \\cite{{key}} or evidence.
- Formal academic tone. No colloquialisms, contractions, or casual phrasing.
- Technical terms defined on first use.

## LATEX RULES
1. Write ONLY the LaTeX body content for each section.
2. Use \\cite{{key}} for citations (only keys listed above).
3. Do NOT include \\documentclass, \\usepackage, \\begin{{document}}, or \\end{{document}}.
4. Do NOT use any custom commands or undefined macros.
5. Only use standard LaTeX commands: \\section, \\subsection, \\textbf, \\textit, \\cite, \\ref, itemize, enumerate, equation, table, tabular, figure environments.
6. Write in full academic prose paragraphs, not bullet points.
7. Escape special characters: use \\% for percent, \\& for ampersand, etc.

## Output Format
Return your content between markers like this:

%%BEGIN TITLE%%
<the paper title — specific, under 15 words>
%%END TITLE%%

%%BEGIN ABSTRACT%%
<abstract text — 150-250 words, C-C-C structure, no \\begin{{abstract}} tags>
%%END ABSTRACT%%

%%BEGIN BODY%%
\\section{{Introduction}}
<introduction: broad context → specific gap → contribution statement>

\\section{{Background}}
<background: synthesize prior work by theme, cite extensively>

... (include only sections listed above)

\\section{{Conclusion}}
<conclusion: concise synthesis, how this advances the field>
%%END BODY%%

%%BEGIN BIBTEX%%
@article{{citekey1,
  author = {{Last, First}},
  title = {{Paper Title}},
  journal = {{Journal Name}},
  year = {{2024}},
  volume = {{1}},
  pages = {{1--10}}
}}
... (one BibTeX entry for EVERY \\cite{{key}} used in the body)
%%END BIBTEX%%

## CRITICAL: BibTeX Requirements
- You MUST include a %%BEGIN BIBTEX%% ... %%END BIBTEX%% section.
- Every \\cite{{key}} in the body MUST have a matching @article/@inproceedings entry in the BIBTEX section.
- Use realistic metadata: real author names, real paper titles, real venues, accurate years.
- Cite keys must match exactly between \\cite{{}} and @type{{key, ...}}.

Write the report now."""
        text, usage = call_llm(prompt)
        return text, usage

    def post(self, shared, prep_res, exec_res):
        text, usage = exec_res
        track_cost(shared, "write_tex", usage)

        # Extract sections from markers
        title_match = re.search(r"%%BEGIN TITLE%%(.*?)%%END TITLE%%", text, re.DOTALL)
        abstract_match = re.search(r"%%BEGIN ABSTRACT%%(.*?)%%END ABSTRACT%%", text, re.DOTALL)
        body_match = re.search(r"%%BEGIN BODY%%(.*?)%%END BODY%%", text, re.DOTALL)
        bibtex_match = re.search(r"%%BEGIN BIBTEX%%(.*?)%%END BIBTEX%%", text, re.DOTALL)

        title = title_match.group(1).strip() if title_match else prep_res["topic"]
        abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not available."
        body = body_match.group(1).strip() if body_match else text.strip()

        # Extract BibTeX entries from WriteTeX output and merge with skill-collected entries
        if bibtex_match:
            bib_block = bibtex_match.group(1).strip()
            tex_bib_entries = re.findall(r"(@\w+\{[^@]+)", bib_block, re.DOTALL)
            tex_bib_entries = [e.strip() for e in tex_bib_entries if e.strip()]
            shared.setdefault("bibtex_entries", []).extend(tex_bib_entries)
            print(f"[WriteTeX] Extracted {len(tex_bib_entries)} BibTeX entries from report")

        # Assemble .tex from skeleton
        tex = LATEX_SKELETON
        tex = tex.replace("%% TITLE %%", title)
        tex = tex.replace("%% ABSTRACT %%", abstract)
        tex = tex.replace("%% BODY %%", body)

        # Deduplicate and write .bib
        bib_content = dedup_bibtex(shared.get("bibtex_entries", []))

        # Use existing task directory (created by BudgetPlanner)
        out_dir = Path(shared["output_path"])

        (out_dir / "report.tex").write_text(tex, encoding="utf-8")
        (out_dir / "references.bib").write_text(bib_content, encoding="utf-8")

        shared["tex_content"] = tex
        shared["bib_content"] = bib_content

        # --- Citation validation ---
        cite_keys_in_tex = set(re.findall(r"\\cite\{([^}]+)\}", body))
        # Expand comma-separated keys like \cite{a,b,c}
        all_cite_keys = set()
        for group in cite_keys_in_tex:
            for key in group.split(","):
                all_cite_keys.add(key.strip())

        bib_keys = set()
        for entry in shared.get("bibtex_entries", []):
            m = re.match(r"@\w+\{([^,]+),", entry)
            if m:
                bib_keys.add(m.group(1).strip())

        missing = all_cite_keys - bib_keys
        if not bib_content.strip():
            print(f"[WriteTeX] WARNING: references.bib is EMPTY — all citations will show as [?]")
        elif missing:
            print(f"[WriteTeX] WARNING: {len(missing)} cite keys missing from .bib: {', '.join(sorted(missing)[:10])}")
        else:
            print(f"[WriteTeX] Citation check passed: {len(all_cite_keys)} keys, all resolved")

        print(f"[WriteTeX] Wrote report.tex + references.bib to {out_dir}")
        print(f"[WriteTeX] Budget remaining: ${shared['budget_remaining']:.4f}")
        return "compile"


# ===================================================================
# 5. CompileTeX
# ===================================================================
class CompileTeX(Node):
    """Compile .tex + .bib → .pdf using pdflatex + bibtex."""

    def prep(self, shared):
        return shared["output_path"]

    def exec(self, out_dir):
        import shutil
        if not shutil.which("pdflatex"):
            return None, "pdflatex not found"

        cmds = [
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            ["bibtex", "report"],
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
        ]
        all_output = []
        for cmd in cmds:
            result = subprocess.run(
                cmd,
                cwd=out_dir,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=60,
            )
            all_output.append(result.stdout + result.stderr)

        # Check if PDF was produced
        pdf_path = Path(out_dir) / "report.pdf"
        success = pdf_path.exists()
        return success, "\n".join(all_output)

    def post(self, shared, prep_res, exec_res):
        success, log = exec_res
        if success is None:
            print(f"[CompileTeX] pdflatex not installed — skipping PDF compilation.")
            print(f"[CompileTeX] LaTeX source ready at: {shared['output_path']}/report.tex")
            print(f"[CompileTeX] To compile manually: cd {shared['output_path']} && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex")
            return "done"

        # Check for undefined citations even if PDF was produced
        undefined_cites = re.findall(r"Citation `([^']+)' on page", log)
        if undefined_cites:
            unique_missing = sorted(set(undefined_cites))
            print(f"[CompileTeX] WARNING: {len(unique_missing)} undefined citations: {', '.join(unique_missing[:10])}")
            shared["has_citation_warnings"] = True

        if success:
            if undefined_cites and shared.get("fix_attempts", 0) < 2:
                print(f"[CompileTeX] PDF has {len(unique_missing)} undefined citations, routing to fix...")
                shared["compile_errors"] = log
                shared["undefined_citations"] = unique_missing
                return "fix"
            elif undefined_cites:
                print(f"[CompileTeX] PDF compiled with citation warnings (fix attempts exhausted): {shared['output_path']}/report.pdf")
            else:
                print(f"[CompileTeX] PDF compiled successfully: {shared['output_path']}/report.pdf")
            return "done"
        else:
            shared["compile_errors"] = log
            print("[CompileTeX] Compilation failed, attempting fix...")
            return "fix"


# ===================================================================
# 6. FixTeX
# ===================================================================
class FixTeX(Node):
    """Fix LaTeX compilation errors or undefined citations."""

    def prep(self, shared):
        undefined_cites = shared.get("undefined_citations", [])
        return {
            "tex_content": shared["tex_content"],
            "bib_content": shared.get("bib_content", ""),
            "errors": shared.get("compile_errors", ""),
            "attempt": shared.get("fix_attempts", 0),
            "undefined_citations": undefined_cites,
            "mode": "citation" if undefined_cites else "latex_error",
        }

    def exec(self, prep_res):
        if prep_res["attempt"] >= 2:
            # Give up after 2 fix attempts
            return None, {"input_tokens": 0, "output_tokens": 0, "cost": 0}

        if prep_res["mode"] == "citation":
            # Citation fix mode: generate missing BibTeX entries
            missing_keys = prep_res["undefined_citations"]
            prompt = f"""The following BibTeX citation keys are used in a LaTeX document with \\cite{{key}} but are missing from the .bib file, causing [?] markers in the PDF.

## Missing Citation Keys
{', '.join(missing_keys)}

## Current .bib Content (for context, do NOT repeat existing entries)
{prep_res["bib_content"][:3000]}

## Instructions
1. For EACH missing key listed above, generate a plausible BibTeX entry.
2. Use the cite key EXACTLY as listed (do not rename it).
3. Use realistic metadata: real author names, real paper titles, real venues.
4. Each entry MUST have: author, title, year, and journal/booktitle.
5. Return ONLY the new BibTeX entries, nothing else. No explanation text.
6. Do not repeat entries already in the .bib file."""
            text, usage = call_llm(prompt)
            return text, usage
        else:
            # LaTeX error fix mode (original behavior)
            error_lines = []
            for line in prep_res["errors"].split("\n"):
                if line.startswith("!") or "Error" in line or "Undefined" in line:
                    error_lines.append(line)
            error_summary = "\n".join(error_lines[:30])

            prompt = f"""Fix these LaTeX compilation errors. Return the COMPLETE corrected .tex file content.

## Errors
{error_summary}

## Current .tex Content
{prep_res["tex_content"]}

## Rules
1. Do NOT change the \\documentclass or \\usepackage lines.
2. Only fix the errors in the body content.
3. Common fixes: escape special chars (%, &, #, $, _), close environments, fix undefined commands.
4. Return ONLY the complete .tex file, nothing else."""
            text, usage = call_llm(prompt)
            return text, usage

    def post(self, shared, prep_res, exec_res):
        text, usage = exec_res

        shared["fix_attempts"] = prep_res["attempt"] + 1

        if text is None:
            # Max attempts reached
            print(f"[FixTeX] Max fix attempts reached. Output may have compilation warnings.")
            return "done"

        track_cost(shared, f"fix_tex:{shared['fix_attempts']}", usage)

        # Clean up: strip markdown fences if the LLM wrapped it
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        if prep_res["mode"] == "citation":
            # Citation fix: parse new entries, validate, and update .bib
            new_entries = re.findall(r"(@\w+\{[^@]+)", cleaned, re.DOTALL)
            new_entries = [e.strip() for e in new_entries if e.strip()]

            if new_entries:
                all_entries = shared.get("bibtex_entries", []) + new_entries
                combined = dedup_bibtex(all_entries)

                out_dir = Path(shared["output_path"])
                (out_dir / "references.bib").write_text(combined, encoding="utf-8")
                shared["bib_content"] = combined
                shared["bibtex_entries"] = all_entries

                print(f"[FixTeX] Added {len(new_entries)} BibTeX entries for undefined citations")

            # Clear flag so CompileTeX re-evaluates from scratch
            shared.pop("undefined_citations", None)
            print(f"[FixTeX] Citation fix applied (attempt {shared['fix_attempts']})")
            return "compile"
        else:
            # LaTeX error fix: rewrite .tex
            out_dir = Path(shared["output_path"])
            (out_dir / "report.tex").write_text(cleaned, encoding="utf-8")
            shared["tex_content"] = cleaned

            print(f"[FixTeX] Applied fix (attempt {shared['fix_attempts']})")
            return "compile"


# ===================================================================
# 7. Finisher — terminal node (no successors → flow ends cleanly)
# ===================================================================
class Finisher(Node):
    """Print cost summary and end the flow."""

    def prep(self, shared):
        return shared

    def exec(self, prep_res):
        return None

    def post(self, shared, prep_res, exec_res):
        import json as _json
        from datetime import datetime, timezone

        total = sum(entry["cost"] for entry in shared.get("cost_log", []))
        out_dir = Path(shared["output_path"])

        # Persist cost log
        (out_dir / "cost_log.json").write_text(
            _json.dumps(shared.get("cost_log", []), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Persist final summary for post-analysis
        summary = {
            "topic": shared.get("topic", ""),
            "domain": shared.get("domain", ""),
            "report_type": shared.get("report_type", ""),
            "budget_dollars": shared.get("budget_dollars", 0),
            "total_cost": round(total, 6),
            "budget_remaining": round(shared.get("budget_remaining", 0), 6),
            "steps_executed": len(shared.get("history", [])),
            "artifacts": list(shared.get("artifacts", {}).keys()),
            "bibtex_count": len(shared.get("bibtex_entries", [])),
            "fix_attempts": shared.get("fix_attempts", 0),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "summary.json").write_text(
            _json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\n{'='*50}")
        print(f"Research complete!")
        print(f"Total cost: ${total:.4f}")
        print(f"Budget used: ${total:.4f} / ${shared['budget_dollars']:.2f}")
        print(f"Output: {shared['output_path']}/")
        print(f"{'='*50}")
