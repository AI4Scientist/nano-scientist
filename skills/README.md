# Skills

This directory contains the **skill library** for Nano-scientist. Each subdirectory is a self-contained skill that the agent can invoke during a research run.

## How It Works

Skills are lazy-loaded at runtime. The agent reads `skills/skills.json` at startup for routing (id + description only), then loads the full `SKILL.md` body only when a skill is selected for execution.

Skills with `allowed-tools: Bash` get a real bash tool-calling loop — the model drives shell execution, sees stdout/stderr, and retries on error (up to `MAX_TOOL_ROUNDS`, default 16).

Skills without `allowed-tools` use a plain LLM call with no tool access.

## Available Skills

87 skills total. Skills inherit from ARIS (Auto-claude-code-research-in-sleep). Only `Bash` tool access is honoured by nano-scientist; other `allowed-tools` values (Agent, Skill, mcp__codex__*, etc.) are present in the SKILL.md body but are not executed.

### Literature & Ideation

| Skill | Description |
| ----- | ----------- |
| [`arxiv`](arxiv/) | Search, download, and summarize arXiv papers |
| [`alphaxiv`](alphaxiv/) | Quick single-paper lookup via AlphaXiv LLM-optimized summaries |
| [`deepxiv`](deepxiv/) | Layered paper access and section-level reading via DeepXiv |
| [`exa-search`](exa-search/) | AI-powered web search via Exa with content extraction |
| [`gemini-search`](gemini-search/) | Research paper search via Gemini for broad literature discovery |
| [`openalex`](openalex/) | Academic papers via OpenAlex API: citation data, affiliations, funding |
| [`paper-navigator`](paper-navigator/) | Find and read papers: search, citation traversal, arXiv monitoring, SOTA |
| [`research-lit`](research-lit/) | Search and analyze research papers, find related work |
| [`research-survey`](research-survey/) | Structured literature survey reports: outline, draft, section expansion, assembly |
| [`semantic-scholar`](semantic-scholar/) | Published venue papers via Semantic Scholar: citation counts, venue metadata |
| [`comm-lit-review`](comm-lit-review/) | Communications-domain literature review |
| [`novelty-check`](novelty-check/) | Verify research idea novelty against recent literature |

### Idea Generation & Refinement

| Skill | Description |
| ----- | ----------- |
| [`idea-creator`](idea-creator/) | Generate and rank research ideas from a broad direction |
| [`idea-discovery`](idea-discovery/) | Full idea discovery pipeline: research-lit → idea-creator → novelty-check → review |
| [`idea-discovery-robot`](idea-discovery-robot/) | Idea discovery adapted for robotics and embodied AI |
| [`research-ideation`](research-ideation/) | End-to-end ideation: literature → multi-persona generation → ELO → proposal |
| [`research-refine`](research-refine/) | Turn vague direction into focused, implementation-oriented method plan |
| [`research-refine-pipeline`](research-refine-pipeline/) | Chain research-refine + experiment-plan in one shot |
| [`research-review`](research-review/) | Deep critical review of research ideas via external LLM |
| [`kill-argument`](kill-argument/) | Two-thread adversarial review: strongest rejection + point-by-point defense |

### Experiment Execution

| Skill | Description |
| ----- | ----------- |
| [`experiment-plan`](experiment-plan/) | Detailed, claim-driven experiment roadmap from a refined proposal |
| [`experiment-pipeline`](experiment-pipeline/) | Structured 4-stage execution: baseline, tuning, validation, ablation |
| [`experiment-craft`](experiment-craft/) | Debug/diagnose existing experiments with structured iteration logging |
| [`experiment-iterative-coder`](experiment-iterative-coder/) | Iterative code refinement via plan→code→evaluate→refine cycles |
| [`experiment-bridge`](experiment-bridge/) | W1.5: Read EXPERIMENT_PLAN.md, implement code, deploy, collect results |
| [`experiment-audit`](experiment-audit/) | Audit experiment integrity: check for fake ground truth, phantom results |
| [`experiment-queue`](experiment-queue/) | SSH job queue for multi-seed/multi-config sweeps with OOM-aware retry |
| [`dse-loop`](dse-loop/) | Autonomous design space exploration for architecture/EDA |
| [`run-experiment`](run-experiment/) | Deploy and run ML experiments on local/remote/Vast.ai/Modal GPU |
| [`serverless-modal`](serverless-modal/) | Run GPU workloads on Modal serverless |
| [`vast-gpu`](vast-gpu/) | Rent and manage GPU instances on vast.ai |
| [`monitor-experiment`](monitor-experiment/) | Monitor running experiments and collect results |
| [`training-check`](training-check/) | Periodically check WandB metrics to catch NaN/divergence early |
| [`qzcli`](qzcli/) | Manage GPU jobs on the Qizhi platform via qzcli |
| [`analyze-results`](analyze-results/) | Analyze ML results, compute statistics, generate comparison tables |
| [`result-to-claim`](result-to-claim/) | Judge what claims experiment results support; route to pivot/supplement/confirm |
| [`ablation-planner`](ablation-planner/) | Design ablation studies from a reviewer's perspective |
| [`system-profile`](system-profile/) | Profile scripts, processes, GPU, memory, interconnect |

### Memory & Knowledge

| Skill | Description |
| ----- | ----------- |
| [`evo-memory`](evo-memory/) | Persistent research memory: M_I + M_E stores via IDE/IVE/ESE mechanisms |
| [`research-wiki`](research-wiki/) | Persistent knowledge base: papers, ideas, experiments, claims, relationships |

### Paper Writing

| Skill | Description |
| ----- | ----------- |
| [`paper-planning`](paper-planning/) | Pre-writing planning: story design, experiment planning, figure design, timeline |
| [`paper-plan`](paper-plan/) | Generate a structured paper outline from review conclusions and results |
| [`paper-write`](paper-write/) | Draft LaTeX paper section by section from an outline |
| [`paper-writing`](paper-writing/) | Full paper writing with 11-step workflow and LaTeX templates |
| [`paper-review`](paper-review/) | Self-review before submission: adversarial stress-testing, 5-aspect checklist |
| [`paper-rebuttal`](paper-rebuttal/) | Peer-review rebuttal: score diagnosis, comment prioritization, champion strategy |
| [`rebuttal`](rebuttal/) | W4: Full submission rebuttal pipeline under venue limits |
| [`paper-figure`](paper-figure/) | Generate publication-quality figures and tables from experiment results |
| [`figure-spec`](figure-spec/) | Deterministic SVG diagrams from structured JSON (FigureSpec) |
| [`mermaid-diagram`](mermaid-diagram/) | Generate Mermaid diagrams (flowcharts, sequence, ER, Gantt, etc.) |
| [`paper-illustration`](paper-illustration/) | AI illustrations for papers via Gemini image generation |
| [`paper-illustration-image2`](paper-illustration-image2/) | AI illustrations via Codex native image generation (experimental) |
| [`figure-description`](figure-description/) | Generate formal patent drawing descriptions from figures |
| [`paper-compile`](paper-compile/) | Compile LaTeX to PDF with error fixing |
| [`paper-slides`](paper-slides/) | Conference slides: Beamer LaTeX → PDF + PPTX with speaker notes |
| [`paper-talk`](paper-talk/) | End-to-end conference talk: paper → slides → assurance checks → export |
| [`paper-poster`](paper-poster/) | Conference poster: LaTeX → A0/A1 PDF + PPTX + SVG |
| [`slides-polish`](slides-polish/) | Per-page Codex review + python-pptx/Beamer fixes for talk slides |
| [`formula-derivation`](formula-derivation/) | Structure and derive research formulas into paper-ready derivations |
| [`proof-writer`](proof-writer/) | Write rigorous mathematical proofs for ML/AI theory |
| [`proof-checker`](proof-checker/) | Verify and fix mathematical proofs via cross-model review |
| [`citation-audit`](citation-audit/) | Zero-context verification of bibliographic entries |
| [`paper-claim-audit`](paper-claim-audit/) | Verify paper numbers/claims against raw result files |
| [`research-pipeline`](research-pipeline/) | Full pipeline: idea discovery → implementation → review loop → paper |
| [`auto-review-loop`](auto-review-loop/) | Autonomous multi-round review loop: review → fix → re-review |
| [`auto-review-loop-llm`](auto-review-loop-llm/) | Review loop using any OpenAI-compatible LLM API |
| [`auto-review-loop-minimax`](auto-review-loop-minimax/) | Review loop using MiniMax API |
| [`auto-paper-improvement-loop`](auto-paper-improvement-loop/) | Autonomously improve a paper: review → fix → recompile (2 rounds) |
| [`resubmit-pipeline`](resubmit-pipeline/) | Resubmit polished paper to a different venue under hard constraints |
| [`overleaf-sync`](overleaf-sync/) | Two-way sync between local paper directory and Overleaf |
| [`study-workflow`](study-workflow/) | Research workflow diagram as PNG via gpt-image-2 |
| [`writing-systems-papers`](writing-systems-papers/) | Structural blueprint for OSDI/SOSP/ASPLOS/NSDI/EuroSys papers |

### Patent

| Skill | Description |
| ----- | ----------- |
| [`patent-pipeline`](patent-pipeline/) | Full patent drafting: invention → claims → spec → jurisdiction format |
| [`claims-drafting`](claims-drafting/) | Draft patent claims for an invention |
| [`specification-writing`](specification-writing/) | Write the full patent specification from claims and disclosure |
| [`embodiment-description`](embodiment-description/) | Write detailed embodiment descriptions for patent specs |
| [`figure-description`](figure-description/) | Generate formal drawing descriptions from patent figures |
| [`invention-structuring`](invention-structuring/) | Structure a raw idea into a formal invention disclosure |
| [`patent-novelty-check`](patent-novelty-check/) | Assess patent novelty and non-obviousness |
| [`prior-art-search`](prior-art-search/) | Search patent databases and literature for prior art |
| [`patent-review`](patent-review/) | External patent examiner review of a patent application |
| [`jurisdiction-format`](jurisdiction-format/) | Compile patent into jurisdiction-specific format (CN/US/EP) |

### Tools & Utilities

| Skill | Description |
| ----- | ----------- |
| [`pixel-art`](pixel-art/) | Generate pixel art SVG illustrations for READMEs/docs/slides |
| [`feishu-notify`](feishu-notify/) | Send notifications to Feishu/Lark (webhook or interactive) |
| [`grant-proposal`](grant-proposal/) | Draft grant proposals (KAKENHI, NSF, NSFC, ERC, etc.) |
| [`meta-optimize`](meta-optimize/) | Analyze usage logs, propose optimizations to SKILL.md files |

## Skill Anatomy

```
my-skill/
  SKILL.md          # required — frontmatter + body
  scripts/          # optional — executable scripts invoked via bash tool
  references/       # optional — docs loaded into agent context
```

## SKILL.md Frontmatter

```yaml
---
name: my-skill
description: "One-line summary used for routing."
allowed-tools: Bash          # grants bash tool-calling loop with error feedback
required-keys: [MY_API_KEY]  # optional; skill filtered out at startup if key missing
metadata:
  author: YourName
  version: '1.0.0'
  tags: [relevant, keywords]
---
```

## Adding a Skill

1. Create `skills/my-skill/SKILL.md` with the frontmatter above and your instructions in the body.
2. Add an entry to `skills/skills.json`:
   ```json
   { "id": "my-skill", "description": "One-line description shown to the agent." }
   ```
3. Validate: `python skills/validate_skills.py`

## Validating Skills

```bash
python skills/validate_skills.py
```

Checks all `skills/*/SKILL.md` files for required frontmatter fields, valid `allowed-tools`, and matching `name` vs directory name.
