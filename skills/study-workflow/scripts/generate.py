#!/usr/bin/env python3
"""Generate a research workflow diagram via the OpenAI image API.

Usage:
    python generate.py --draft PATH_OR_TEXT [--output PATH]
"""
import argparse
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _build_prompt(draft: str) -> str:
    return f"""You are a scientific visualization expert. Read the paper draft below and produce a publication-quality workflow figure in the style of top-tier AI/ML conference papers (NeurIPS, ACL, CVPR, ICLR).

---

PAPER DRAFT:
{draft}

---

## Global Layout Constraints (STRICT)

- The figure must span fully from the left edge to the right edge of the canvas
- Use the full horizontal width; do NOT compress into a centered block
- All stages must be arranged in a single horizontal row (left → right)
- Maintain generous vertical whitespace above and below (top/bottom margins ≥10%)
- Vertically center all content; no element may touch top/bottom margins
- Keep inter-stage spacing uniform and visually balanced

---

## Stage Panel Design

- Each stage is a rounded rectangle with:
  - **solid border (MANDATORY; never dashed)**
  - light tinted background (low saturation)
  - distinct accent color per stage (blue / orange / green / purple / teal)

- Header (top of each panel):
  - Format: "Stage X  TITLE"
  - Title in ALL CAPS, bold
  - Accent color applied to title and border
  - Optional count in parentheses allowed (e.g., "(56)") — but NO explanations of what it means

- Subtitle (1 short line under header):
  - concise phrase (≤8 words)
  - no full sentences

---

## Internal Structure (Within Each Stage)

Each stage must follow a **3-tier hierarchy**:

1. **Step Row (horizontal mini-boxes)**
   - Small rounded boxes labeled A, B, C...
   - Each contains a short phrase only
   - Solid borders only

2. **Strategy Blocks (main content boxes)**
   - Larger rounded rectangles below steps
   - Bullet-style entries with:
     - short label
     - optional numeric value
     - optional small icon (inline, right-aligned)

3. **NO verbose text**
   - Use phrases only
   - No explanatory sentences
   - No annotations like “numbers indicate …”

---

## Arrows & Flow Semantics

- Primary pipeline flow:
  - Thick **solid horizontal arrows** between stages
  - Positioned at vertical centerline

- Secondary relations (if needed):
  - Thin **dashed arrows**
  - ONLY for intra-stage or vertical linking
  - NEVER use dashed borders for boxes

---

## Legend (Bottom Row — STRICT FORMAT)

- Place a **single horizontal legend bar centered at the bottom**
- Must contain exactly three items:
  - Offline / Local
  - External / Online
  - Live / Production

- Layout:
  - Icons + labels arranged horizontally
  - Even spacing across width
  - Minimal, clean styling

- DO NOT include:
  - Any explanatory sentences
  - Any notes about counts or percentages

---

## Explicitly Forbidden Elements

- ❌ No dashed borders on any boxes
- ❌ No percentage callouts (e.g., "41.1% issues")
- ❌ No explanatory footnotes (e.g., "numbers in parentheses indicate...")
- ❌ No outer frame or enclosing border
- ❌ No title outside the panels
- ❌ No watermark or attribution
- ❌ No gradients, shadows, or glow effects

---

## Typography & Rendering

- Use crisp, high-resolution vector-like rendering
- Strong hierarchy:
  - Stage title: bold, largest
  - Step labels: medium bold
  - Content: regular
- Align text and icons cleanly; avoid clutter
- Ensure consistent spacing across all panels

---

## Output Requirement

- Output the **figure only**
- No captions, no explanations, no surrounding text
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate research workflow diagram via OpenAI image API")
    ap.add_argument("--output", default="workflow.png", help="Output PNG path (default: workflow.png in cwd)")
    ap.add_argument("--draft", required=True, help="Paper draft — file path or raw text")
    args = ap.parse_args()

    draft_path = Path(args.draft)
    if draft_path.is_file():
        draft = draft_path.read_text(encoding="utf-8")
    else:
        draft = args.draft

    if not draft.strip():
        print("error: --draft is empty", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env", override=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    prompt = _build_prompt(draft)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    try:
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size="auto",
            quality="high",
            n=1,
        )
        b64 = response.data[0].b64_json
        if not b64:
            print(f"error: no image in response — {response}", file=sys.stderr)
            sys.exit(1)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"error: unexpected response shape — {e}: {str(response)[:300]}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(b64))

    print(args.output)


if __name__ == "__main__":
    main()
