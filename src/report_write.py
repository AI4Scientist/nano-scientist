"""report_write.py - Stage 3: Report Generation module

Tectonic-based PDF generation for research reports with citations and figures.
Synthesizes research plan, execution results, and citations into a professional PDF.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import litellm
from config import get_config, ModelConfig


def create_pdf_report(
    plan_file: str,
    citations_file: str,
    results_file: str,
    output_dir: str,
    model_id: str = "anthropic/claude-haiku-4-5-20251001"
) -> str:
    """Generate a PDF research report from plan, citations, and results.

    Args:
        plan_file: Path to research_plan.json
        citations_file: Path to citations.json
        results_file: Path to results.json
        output_dir: Directory for PDF output
        model_id: LLM model for paper synthesis

    Returns:
        Path to generated PDF file
    """
    # Load data
    with open(plan_file, 'r') as f:
        plan = json.load(f)

    with open(citations_file, 'r') as f:
        citations = json.load(f)

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Synthesize paper content using LLM
    paper_content = _synthesize_paper_content(plan, results, citations, model_id)

    # Generate PDF
    pdf_path = _generate_pdf(paper_content, citations, results, output_dir)

    return pdf_path


def _synthesize_paper_content(
    plan: Dict,
    results: Dict,
    citations: List[Dict],
    model_id: str
) -> Dict:
    """Use LLM to synthesize research paper structure from data.

    Args:
        plan: Research plan dictionary
        results: Execution results dictionary
        citations: List of citation dictionaries
        model_id: LLM model to use

    Returns:
        Dictionary with paper structure (title, authors, abstract, sections)
    """
    config = get_config()

    # Normalize model_id
    if model_id:
        # Use provided model_id but normalize it
        temp_config = ModelConfig(model_id=model_id)
        normalized_model = temp_config.normalize_model_id()
    else:
        normalized_model = config.stage3_model.normalize_model_id()

    # Create citation reference list with BibTeX keys
    citation_refs = [
        {
            "id": c["id"],
            "bibtex_key": c.get("bibtex_key", f"ref{c['id']}"),
            "title": c["title"],
            "authors": c.get("authors", "Unknown"),
            "year": c.get("year", "n.d."),
            "url": c.get("url", "")
        }
        for c in citations
    ]

    prompt = f"""Synthesize a research paper from this data:

RESEARCH PLAN:
{json.dumps(plan, indent=2)}

EXPERIMENTAL RESULTS:
{json.dumps(results, indent=2)}

CITATIONS AVAILABLE:
{json.dumps(citation_refs, indent=2)}

Generate a JSON structure for an academic research paper. Use LaTeX citation commands like \\cite{{ref1}}, \\cite{{ref2}} throughout the text where appropriate. You can also use \\citep{{}} for parenthetical citations or multiple citations like \\cite{{ref1,ref2}}.

The BibTeX keys are provided in the citations list above (use the 'bibtex_key' field).

Required JSON structure:
{{
  "title": "Concise paper title (max 100 characters)",
  "authors": "Author name(s)",
  "abstract": "Abstract paragraph with key findings and citations \\cite{{ref1}}\\cite{{ref2}}",
  "sections": [
    {{
      "title": "Introduction",
      "content": "Introduce the research topic, motivation, and objectives. Include relevant citations using \\cite{{refX}}."
    }},
    {{
      "title": "Methodology",
      "content": "Describe the experimental approach, tools used, and procedures followed."
    }},
    {{
      "title": "Results",
      "content": "Present the findings with specific metrics and data from the experiments. Reference figures where applicable."
    }},
    {{
      "title": "Discussion",
      "content": "Interpret the results, discuss implications, and relate to existing work with citations like \\cite{{refX}}."
    }},
    {{
      "title": "Conclusion",
      "content": "Summarize key findings and future work."
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""

    # Make LiteLLM call
    response = litellm.completion(
        model=normalized_model,
        messages=[{'role': 'user', 'content': prompt}],
        response_format={"type": "json_object"},
        temperature=config.stage3_model.temperature
    )

    content = response.choices[0].message.content

    # Parse JSON response
    try:
        paper = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            paper = json.loads(json_match.group())
        else:
            # Fallback structure
            paper = {
                "title": plan.get("task", "Research Report"),
                "authors": "AI Researcher",
                "abstract": "This research explores " + plan.get("task", "the given topic"),
                "sections": [
                    {"title": "Introduction", "content": f"Research on: {plan.get('task', 'N/A')}"},
                    {"title": "Methodology", "content": plan.get("methodology", {}).get("approach", "N/A")},
                    {"title": "Results", "content": "\n".join(results.get("findings", ["No findings"]))},
                    {"title": "Discussion", "content": "Analysis of results."},
                    {"title": "Conclusion", "content": "Summary of findings."}
                ]
            }

    return paper


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text, preserving citation commands.

    Args:
        text: Raw text to escape

    Returns:
        LaTeX-safe text
    """
    import re

    # Protect citation commands by temporarily replacing them
    citations = []
    def save_citation(match):
        citations.append(match.group(0))
        return f"__CITATION_{len(citations)-1}__"

    # Find and save all citation commands
    text = re.sub(r'\\cite[tp]?\{[^}]+\}', save_citation, text)

    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }

    # Apply replacements
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)

    # Restore citation commands
    for i, citation in enumerate(citations):
        result = result.replace(f"__CITATION_{i}__", citation)

    return result


def _generate_bibtex_entry(citation: Dict) -> str:
    """Generate a BibTeX entry from citation metadata.

    Args:
        citation: Citation dictionary with metadata

    Returns:
        BibTeX entry as string
    """
    bibtex_key = citation.get("bibtex_key", f"ref{citation['id']}")
    entry_type = citation.get("type", "misc")

    # Start entry
    lines = [f"@{entry_type}{{{bibtex_key},"]

    # Add title
    title = citation.get("title", "Unknown Title")
    lines.append(f"  title = {{{title}}},")

    # Add authors
    authors = citation.get("authors", "Unknown")
    if authors and authors != "Unknown":
        lines.append(f"  author = {{{authors}}},")

    # Add year
    year = citation.get("year", "n.d.")
    if year and year != "n.d.":
        lines.append(f"  year = {{{year}}},")

    # Add journal if present
    journal = citation.get("journal", "")
    if journal:
        lines.append(f"  journal = {{{journal}}},")

    # Add publisher if present
    publisher = citation.get("publisher", "")
    if publisher:
        lines.append(f"  publisher = {{{publisher}}},")

    # Add URL
    url = citation.get("url", "")
    if url:
        lines.append(f"  url = {{{url}}},")
        lines.append(f"  note = {{Accessed: {{}}}},")

    # Close entry (remove trailing comma from last line)
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")

    return "\n".join(lines)


def _generate_bibtex_file(citations: List[Dict], output_path: Path) -> str:
    """Generate a BibTeX file from citations.

    Args:
        citations: List of citation dictionaries
        output_path: Directory for output

    Returns:
        Path to generated .bib file
    """
    bib_content = "% BibTeX bibliography generated by Mini-Researcher-Agent\n\n"

    for citation in citations:
        bib_content += _generate_bibtex_entry(citation) + "\n\n"

    bib_file = output_path / "references.bib"
    with open(bib_file, 'w', encoding='utf-8') as f:
        f.write(bib_content)

    return str(bib_file)


def _generate_latex_content(
    paper_content: Dict,
    citations: List[Dict],
    results: Dict,
    bib_filename: str = "references"
) -> str:
    """Generate LaTeX source code for the paper using BibTeX.

    Args:
        paper_content: Paper structure from synthesis
        citations: List of citations
        results: Execution results (for figures)
        bib_filename: Name of BibTeX file (without .bib extension)

    Returns:
        LaTeX source code as string
    """
    title = _escape_latex(paper_content.get("title", "Research Report"))
    authors = _escape_latex(paper_content.get("authors", "AI Researcher"))
    abstract = _escape_latex(paper_content.get("abstract", ""))

    # Start building LaTeX document with natbib for citations
    latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage[numbers,sort&compress]{natbib}
\geometry{margin=1in}

\title{""" + title + r"""}
\author{""" + authors + r"""}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
""" + abstract + r"""
\end{abstract}

"""

    # Add main sections
    sections = paper_content.get("sections", [])
    for sec in sections:
        section_title = _escape_latex(sec.get("title", "Untitled"))
        section_content = _escape_latex(sec.get("content", ""))

        latex += f"\\section{{{section_title}}}\n\n"
        latex += section_content + "\n\n"

        # If this is the Results section, try to embed figures
        if sec.get("title") == "Results":
            figures = results.get("figures", [])
            for fig_data in figures:
                fig_path = Path(fig_data.get("path", ""))
                if fig_path.exists():
                    caption = _escape_latex(fig_data.get("caption", "Figure"))
                    latex += f"""
\\begin{{figure}}[h!]
    \\centering
    \\includegraphics[width=0.7\\textwidth]{{{fig_path}}}
    \\caption{{{caption}}}
\\end{{figure}}

"""

    # Add bibliography using BibTeX
    if citations:
        latex += r"""
% Bibliography
\bibliographystyle{unsrt}
\bibliography{""" + bib_filename + r"""}

"""

    latex += r"""\end{document}
"""

    return latex


def _generate_pdf(
    paper_content: Dict,
    citations: List[Dict],
    results: Dict,
    output_dir: str
) -> str:
    """Generate PDF using Tectonic with BibTeX support.

    Args:
        paper_content: Paper structure from synthesis
        citations: List of citations
        results: Execution results (for figures)
        output_dir: Output directory

    Returns:
        Path to generated PDF
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate BibTeX file
    if citations:
        bib_file = _generate_bibtex_file(citations, output_path)
        print(f"✓ BibTeX file generated: {bib_file}")
        bib_filename = "references"  # Without .bib extension
    else:
        bib_filename = "references"

    # Generate LaTeX content
    latex_content = _generate_latex_content(paper_content, citations, results, bib_filename)

    # Generate filename
    title_safe = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in paper_content.get("title", "report"))
    title_safe = title_safe.replace(' ', '_')[:50]
    tex_file = output_path / f"paper_{title_safe}.tex"
    pdf_file = output_path / f"paper_{title_safe}.pdf"

    # Write LaTeX file
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    # Compile with Tectonic
    try:
        # Add ~/.local/bin to PATH in case tectonic is installed there
        env = os.environ.copy()
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

        result = subprocess.run(
            ['tectonic', tex_file.name],
            cwd=output_path,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

        if pdf_file.exists():
            print(f"✓ PDF generated successfully: {pdf_file}")
            return str(pdf_file)
        else:
            raise RuntimeError("Tectonic completed but PDF not found")

    except subprocess.CalledProcessError as e:
        error_msg = f"Tectonic compilation failed:\n{e.stderr}"
        print(f"Warning: {error_msg}")
        print(f"LaTeX source saved to: {tex_file}")
        return str(tex_file)
    except FileNotFoundError:
        error_msg = "Tectonic not found. Please install: pip install tectonic"
        print(f"Warning: {error_msg}")
        print(f"LaTeX source saved to: {tex_file}")
        return str(tex_file)


def validate_report(pdf_path: str) -> tuple[bool, str]:
    """Validate generated PDF report.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (is_valid, error_message)
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        return False, "PDF file does not exist"

    # Check file size (should be >1KB for a real PDF)
    if pdf_file.suffix == '.pdf' and pdf_file.stat().st_size < 1024:
        return False, "PDF file is too small (likely empty)"

    # Basic validation passed
    return True, "Valid"


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python report_write.py <plan_file> <citations_file> <results_file> <output_dir>")
        sys.exit(1)

    plan_file = sys.argv[1]
    citations_file = sys.argv[2]
    results_file = sys.argv[3]
    output_dir = sys.argv[4]

    print(f"Generating PDF report...")
    print(f"  Plan: {plan_file}")
    print(f"  Citations: {citations_file}")
    print(f"  Results: {results_file}")
    print(f"  Output: {output_dir}")

    pdf_path = create_pdf_report(plan_file, citations_file, results_file, output_dir)

    print(f"\n{'='*60}")
    print(f"✓ PDF generated: {pdf_path}")
    print(f"{'='*60}\n")

    # Validate
    is_valid, msg = validate_report(pdf_path)
    if is_valid:
        print("✓ Report validated successfully")
    else:
        print(f"✗ Validation failed: {msg}")
