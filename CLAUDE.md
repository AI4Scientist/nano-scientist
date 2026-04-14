# CLAUDE.md

## Project
Autonomous research agent. Takes a topic + dollar budget, runs a skill loop, outputs a compiled PDF paper.

## Run
```bash
python main.py "topic" --budget 1.00
python main.py --list-skills
```

## Architecture
- **Entry**: `main.py` → `src/flow.py` → `src/nodes.py`
- **Flow**: `BudgetPlanner → DecideNext ↔ ExecuteSkill → WriteTeX → CompileTeX ↔ FixTeX → Finisher`
- **Skills**: `skills/<name>/SKILL.md` — lazy-loaded by `ExecuteSkill`; index in `skills/skills.json`
- **LLM**: OpenRouter via `src/utils.py:call_llm()` — model `z-ai/glm-5`, ~$0.005/call
- **Budget guard**: `BUDGET_RESERVE = 0.03` in `nodes.py` — reserved for final report

## Key files
| File | Role |
|---|---|
| `src/nodes.py` | All 7 agent nodes |
| `src/flow.py` | PocketFlow wiring |
| `src/utils.py` | LLM client, cost tracking, BibTeX utils |
| `skills/skills.json` | Skill index (id + description) |
| `docs/PAPER_QUALITY_STANDARD.md` | Writing quality guide injected into prompts |

## Shared store keys
`topic`, `budget_dollars`, `budget_remaining`, `cost_log`, `skill_index`, `skills_dir`, `output_dir`, `output_path`, `plan`, `history`, `artifacts`, `bibtex_entries`, `generated_files`, `decisions`, `report_type`, `domain`, `quality_standard`, `api_keys`

## Adding a skill
1. Create `skills/<name>/SKILL.md` with YAML frontmatter (`id`, `description`, optionally `allowed-tools: Bash`)
2. Add `{"id": "<name>", "description": "..."}` to `skills/skills.json`

## Environment
Required: `OPENROUTER_API_KEY`. Optional: `PERPLEXITY_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `GITHUB_TOKEN`, `GITLAB_TOKEN`. All in `.env`.

## Conventions
- Skills missing required API keys are auto-excluded from the plan
- Skill code blocks (`%%BEGIN CODE:python%%...%%END CODE%%`) execute in the task output directory
- BibTeX collected per-skill via `%%BEGIN BIBTEX%%...%%END BIBTEX%%`, deduplicated by `dedup_bibtex()`
- Output goes to `outputs/<uuid>/` — do not commit this directory
