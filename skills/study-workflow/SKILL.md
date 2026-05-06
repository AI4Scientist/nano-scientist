---
name: study-workflow
description: Generates a publication-quality research workflow diagram as a PNG via gpt-image-2 (OpenAI image API).
required-keys: [OPENAI_API_KEY]
allowed-tools: Bash
---

Run the generator script with the paper draft as input:

```bash
python skills/study-workflow/scripts/generate.py --draft PATH [--output PATH]
```

- `--draft`: path to a `.tex` / `.md` file, or raw text describing the workflow
- `--output`: output PNG path (default: `workflow.png` in cwd)
