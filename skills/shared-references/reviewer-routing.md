# Reviewer Routing

## Default (NEVER changes without explicit user request)

All review calls use **Codex MCP** (`mcp__codex__codex`) with `reasoning_effort: xhigh`.

This is the default for ALL skills. No parameter, no config, no effort level changes this.

## Optional: GPT-5.4 Pro via Oracle

When the user explicitly passes `— reviewer: oracle-pro`, route the review through Oracle MCP instead of Codex MCP.

### Routing Logic (add to any reviewer-invoking skill)

```
Parse $ARGUMENTS for `— reviewer:` directive.

If not specified OR `— reviewer: codex`:
    → Use mcp__codex__codex with reasoning_effort: xhigh
    → This is the DEFAULT. No change from current behavior.

If `— reviewer: oracle-pro`:
    → Check if mcp__oracle__consult tool is available
    → If available:
        Use mcp__oracle__consult with:
          model: "gpt-5.4-pro"
          prompt: [same prompt you would send to Codex]
          files: [file paths for reviewer to read directly]
        Note: Oracle may use API mode (fast, needs OPENAI_API_KEY)
              or browser mode (slow ~1-2 min, needs Chrome + ChatGPT login)
    → If NOT available:
        Print: "⚠️ Oracle MCP not installed. Falling back to Codex xhigh."
        Use mcp__codex__codex as normal.
```

### Invariants

- `— reviewer: oracle-pro` ONLY takes effect when explicitly passed
- Reviewer independence protocol still applies (pass file paths, not summaries)
- `effort` and `difficulty` are orthogonal — they don't change reviewer backend
- `beast` mode may RECOMMEND oracle-pro but never requires it
- Browser mode: acceptable for one-shot reviews; NOT recommended inside multi-round loops (too slow/brittle)

### Oracle MCP Call Format

```
mcp__oracle__consult:
  prompt: |
    [role + task + output schema]
    Read all listed files directly.
  model: "gpt-5.4-pro"
  files:
    - /absolute/path/to/file1
    - /absolute/path/to/file2
```

### Skills That Support `— reviewer: oracle-pro`

| Skill | Use case for Pro |
|-------|-----------------|
| `/research-review` | Deeper critique on paper drafts |
| `/auto-review-loop` | Final stress test (last round only in browser mode) |
| `/experiment-audit` | Line-by-line eval code audit |
| `/proof-checker` | Deep mathematical reasoning |
| `/rebuttal` | Stress test before submission |
| `/idea-creator` | Idea evaluation depth |
| `/research-lit` | Literature analysis depth |

### Installation

```bash
# Install Oracle CLI + MCP
npm install -g @steipete/oracle

# Add Oracle MCP to Claude Code
claude mcp add oracle -s user -- oracle-mcp

# Restart Claude Code session to load

# API mode (fast, recommended):
export OPENAI_API_KEY="your-key"

# Browser mode (no API key, slower):
# Just log in to ChatGPT in Chrome
```

### NOT installed = ZERO impact

If Oracle is not installed, `— reviewer: oracle-pro` gracefully falls back to Codex. No error, no breakage, just a warning.

### Upstream development & known issues

Oracle MCP is maintained at [`steipete/oracle`](https://github.com/steipete/oracle). When you invoke `— reviewer: oracle-pro` (and especially the `o3-deep-research` / `gpt-5.5-pro` paths), it's worth checking the **[open PRs](https://github.com/steipete/oracle/pulls)** for in-flight fixes that may affect your run — e.g., model routing changes, browser-mode auth fixes, rate-limit handling, or new model alias support. ARIS does not vendor Oracle MCP; you're running the published version from `npm install -g @steipete/oracle`. If a behavior surprises you, the upstream PR queue is the first place to check before opening an issue here.
