---
name: paper-writing
description: Use when writing CS papers for venues like NeurIPS, ICML, ICLR, ACL, AAAI, OSDI, CHI. Synthesizes papers from literature surveys, implementation artifacts, and experimental results. Verifies all citations (no hallucinations), generates reproducibility artifacts, and produces publication-ready LaTeX/PDF.
user-invocable: true
allowed-tools: [Read, Write, Edit, Bash, WebSearch, Task]
version: 1.0.0
license: MIT
tags: [CS Research, Paper Writing, NeurIPS, ICML, ICLR, ACL, LaTeX, Academic Publishing, Citations]
dependencies: [litellm, requests, bibtexparser, pylatex, beautifulsoup4]
---

# CS Paper Writing for Top Venues

## Overview

Generate publication-ready CS papers for top-tier venues (NeurIPS, ICML, ICLR, ACL, AAAI, OSDI, CHI) by synthesizing content from research artifacts: literature surveys, implementations, and experimental evaluations.

**Core Principle**: Produce complete, technically accurate drafts that serve as strong starting points for human refinement.

**Critical Rule**: **NEVER hallucinate citations.** Every reference must be verified programmatically.

---

## Supported Venues

### Machine Learning
- **NeurIPS**: Neural Information Processing Systems (A*)
- **ICML**: International Conference on Machine Learning (A*)
- **ICLR**: International Conference on Learning Representations (A*)
- **AISTATS**: AI and Statistics (A)

### Natural Language Processing
- **ACL**: Association for Computational Linguistics (A*)
- **EMNLP**: Empirical Methods in NLP (A*)
- **NAACL**: North American Chapter of ACL (A)
- **COLM**: Conference on Language Modeling

### Computer Vision
- **CVPR**: Computer Vision and Pattern Recognition (A*)
- **ICCV**: International Conference on Computer Vision (A*)
- **ECCV**: European Conference on Computer Vision (A*)

### Systems
- **OSDI**: Operating Systems Design and Implementation (A*)
- **SOSP**: Symposium on Operating Systems Principles (A*)
- **NSDI**: Networked Systems Design (A*)
- **ATC**: USENIX Annual Technical Conference (A)

### Human-Computer Interaction
- **CHI**: Computer-Human Interaction (A*)
- **UIST**: User Interface Software and Technology (A*)

---

## Workflow

```dot
digraph paper_writing {
    rankdir=TB;
    node [shape=box];

    artifacts [label="Input Artifacts:\n- literature_review.md\n- workspace/\n- evaluation_report.md" shape=parallelogram];
    venue [label="Select Venue\n& Template"];
    synthesize [label="LLM Synthesis\n(Section by Section)"];
    citations [label="Verify Citations\n(No Hallucinations)"];
    latex [label="Generate LaTeX\n(Fixed Processing)"];
    compile [label="Compile PDF"];
    artifacts_check [label="Reproducibility\nArtifact Checklist"];
    end [label="Publication-Ready\nPaper + Artifacts" shape=doublecircle];

    artifacts -> venue;
    venue -> synthesize;
    synthesize -> citations;
    citations -> latex;
    latex -> compile;
    compile -> artifacts_check;
    artifacts_check -> end;
}
```

### Phase 1: Artifact Collection

**Read Input Files**:
```python
# Literature survey
literature = read_markdown("literature_review.md")
related_work = extract_related_work(literature)
citations = extract_citations(literature)

# Implementation
workspace = read_workspace("workspace/")
method_description = extract_method(workspace / "README.md")
code_url = get_repo_url(workspace)

# Evaluation
evaluation = read_markdown("evaluation_report.md")
results_tables = extract_tables(evaluation)
figures = list((evaluation.parent / "figures").glob("*.png"))
```

### Phase 2: Venue Selection & Template

**User Selects Venue**:
```bash
/paper-writing --venue neurips --from-artifacts ./
```

**Load Template**:
```python
template = load_template(venue="neurips")
# Returns: LaTeX template, page limits, formatting rules
```

**Venue-Specific Requirements**:
| Venue | Page Limit | Anonymity | Supplementary | Code Required |
|-------|------------|-----------|---------------|---------------|
| NeurIPS | 9 + refs | Yes (double-blind) | Unlimited | Encouraged |
| ICML | 8 + refs | Yes | Unlimited | Encouraged |
| ACL | 8 + refs | Yes | 8 pages | Yes (for reproducibility) |
| OSDI | 12 + refs | No | Unlimited | Required |

### Phase 3: LLM-Powered Synthesis

**Enhanced Prompt**:
```python
prompt = f"""You are an expert academic researcher writing a {venue} paper.

CRITICAL REQUIREMENTS:
1. **No Citation Hallucinations**: Only use citations from the provided list
2. **Technical Accuracy**: Match method descriptions to implementation
3. **Quantitative Results**: Include all experimental numbers
4. **Complexity Analysis**: Provide Big-O notation where relevant
5. **Code Availability**: Mention reproducibility artifacts

WRITING QUALITY REQUIREMENTS (MANDATORY — read carefully):

You are writing a SCIENTIFIC PAPER, not a technical report or user manual.
Every section must read as connected, argumentative prose.

**Prose-First Rule**: Write in flowing paragraphs. Each paragraph should
have a topic sentence, develop an argument, and transition to the next.
Bullet points and numbered lists are FORBIDDEN except in these cases:
  - A "Contributions" list in the Introduction (max 3-4 items)
  - Algorithm pseudocode (use \\begin{{algorithm}} environment)
  - Items in a formal table
Lists must NEVER replace paragraph prose for explanations, observations,
interpretations, discussions, or proof steps.

**Narrative Flow**: The paper must tell a coherent story:
  - Introduction: Set up the question, build tension ("Can we do better?"),
    then resolve it with your contribution. End with a brief roadmap sentence,
    NOT a subsection called "Paper Organization."
  - Each section must open by connecting to what came before and close by
    motivating what comes next.
  - Avoid mechanical transitions ("The remainder of this paper is organized
    as follows. Section 2 does X. Section 3 does Y."). Instead, weave the
    roadmap into the narrative naturally.

**Proof Writing**: Mathematical proofs MUST be written as continuous
reasoning in paragraph form, not as numbered step lists. Use words like
"observe that," "it follows that," "since," "by combining" to connect
logical steps. Reserve \\begin{{enumerate}} for definitions with multiple
formal conditions, not for proof steps.

**Result Interpretation**: When discussing experimental results, do NOT
produce a numbered list of "Key Observations." Instead, write paragraphs
that interpret the data, explain WHY results look the way they do, connect
findings to the theoretical claims, and note surprises or confirmations.
Weave table/figure references into the prose naturally.

**Discussion Section**: Must synthesize and argue, not list. Each paragraph
should develop one insight in depth — its implication, its connection to
prior work, its limitations. Never write a subsection that is just a
bulleted list of points.

**Section Naming**: Use descriptive names that convey content. Avoid
generic headings like "Main Results" or "Implications." Prefer headings
that hint at the finding, e.g., "MergeSort Achieves the Information-
Theoretic Bound" or "Limitations of the Comparison Model."

**Sentences and Paragraphs**:
  - Vary sentence length. Mix short declarative sentences with longer
    ones that develop nuance.
  - Each paragraph should be 4-8 sentences. Single-sentence paragraphs
    are almost always wrong in a paper.
  - Avoid starting consecutive sentences with the same word.
  - Do not use bold text inside running prose to label sub-points
    (e.g., "\\textbf{{Tightness}}: MergeSort achieves..."). This is
    a disguised list. Write a real paragraph instead.

**What NOT to Write**:
  - "The significance of this result cannot be overstated:" followed by
    a list. → Write a paragraph explaining WHY it matters.
  - "This paper makes the following contributions:" followed by 5 items
    with bold labels. → Keep the list short (3-4 items), one line each,
    no bold-label-colon pattern.
  - "Key Observations:" followed by numbered items. → Write paragraphs.
  - Subsections that contain only a single list and nothing else.
  - "Future Work" as a numbered list of topics. → Write a paragraph
    discussing the most promising 2-3 directions with reasoning.

INPUT ARTIFACTS:
- Literature Survey: {related_work[:3000]}
- Method Implementation: {method_description}
- Experimental Results: {results_summary}
- Available Citations: {citation_keys}

PAPER STRUCTURE (for {venue}):

Abstract (250 words): State the problem, the approach, and the key
quantitative result — all in a single flowing paragraph.

Introduction: Motivate the problem, situate it in the literature, state
your contribution clearly, and preview the paper structure in one sentence.

Related Work: Group by intellectual themes. For each theme, write a
paragraph synthesizing multiple papers and explaining how your work
differs. Do not list papers one by one.

Method: Describe the approach as a narrative. Use formal definitions,
theorems, and proofs in paragraph form. Include pseudocode via the
algorithm environment when helpful, but surround it with explanatory
prose.

Experiments: Describe setup in prose. Present results with tables and
figures, but interpret them in paragraphs — do not just list observations.

Discussion: Synthesize insights, connect back to the introduction's
question, discuss limitations honestly in prose paragraphs.

Conclusion: A concise paragraph summarizing the contribution and its
significance. No lists.

OUTPUT FORMAT:
Generate valid JSON with sections:
{{
  "title": "Concise Paper Title",
  "abstract": "...",
  "sections": [
    {{"name": "Introduction", "content": "..."}},
    ...
  ]
}}

IMPORTANT:
- Use \\cite{{refX}} for citations (keys: {citation_keys})
- Include \\begin{{algorithm}} blocks for pseudocode
- Reference figures: \\ref{{fig:results}}
- Reference tables: \\ref{{tab:comparison}}
- NO markdown formatting (use LaTeX commands)
- NO \\begin{{itemize}} or \\begin{{enumerate}} except for the contributions
  list in the introduction and formal definitions
"""
```

**LLM Call**:
```python
response = litellm.completion(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json"}
)

paper_content = json.loads(response.choices[0].message.content)
```

### Phase 4: Citation Verification

**CRITICAL: Prevent Hallucinated Citations**

**Step 1: Extract Citation Keys**:
```python
cited_keys = re.findall(r'\\cite\{([^}]+)\}', paper_content)
# Returns: ['ref1', 'ref2', 'authors2024', ...]
```

**Step 2: Verify Each Citation**:
```python
for key in cited_keys:
    if key not in verified_citations:
        # Try to find citation
        paper = search_citation(key)
        if paper:
            verified_citations[key] = generate_bibtex(paper)
        else:
            # Mark as placeholder
            paper_content = paper_content.replace(
                f"\\cite{{{key}}}",
                f"\\cite{{PLACEHOLDER_{key}_VERIFY_THIS}}"
            )
            warnings.append(f"Citation {key} not verified")
```

**Step 3: Generate BibTeX**:
```python
bibtex_content = ""
for key, citation in verified_citations.items():
    bibtex_content += citation + "\n\n"

write_file("paper/references.bib", bibtex_content)
```

### Phase 5: LaTeX Generation (Fixed from Stage 3)

**Problem in Old Stage 3**: Markdown formatting leaked into LaTeX
```latex
% BAD (old Stage 3):
\\section{Introduction}
\\# Summary: Finding Rectangular Regions
\\#\\# Overview
```

**Solution**: Proper text processing pipeline

**Step 1: Protect LaTeX Commands**:
```python
def protect_latex(text):
    """Protect existing LaTeX commands from escaping."""
    protected = []
    def protect(match):
        idx = len(protected)
        protected.append(match.group(0))
        return f"__LATEX_{idx}__"

    # Protect: \\cite{}, \\ref{}, $...$, \\begin{}, \\end{}
    patterns = [
        r'\\cite\{[^}]+\}',
        r'\\ref\{[^}]+\}',
        r'\$[^$]+\$',
        r'\\begin\{[^}]+\}',
        r'\\end\{[^}]+\}'
    ]
    for pattern in patterns:
        text = re.sub(pattern, protect, text)

    return text, protected
```

**Step 2: Escape Special Characters**:
```python
def escape_latex_special(text):
    """Escape LaTeX special characters."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}'
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text
```

**Step 3: Convert Markdown to LaTeX**:
```python
def markdown_to_latex(text):
    """Convert markdown formatting to LaTeX."""
    # **bold** -> \\textbf{bold}
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)

    # *italic* -> \\textit{italic}
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)

    # `code` -> \\texttt{code}
    text = re.sub(r'`(.+?)`', r'\\texttt{\1}', text)

    # Lists
    text = convert_lists_to_latex(text)

    return text
```

**Step 4: Restore Protected Commands**:
```python
def restore_latex(text, protected):
    """Restore protected LaTeX commands."""
    for idx, cmd in enumerate(protected):
        text = text.replace(f"__LATEX_{idx}__", cmd)
    return text
```

**Complete Pipeline**:
```python
def prepare_text_for_latex(text):
    """Complete text processing pipeline."""
    text, protected = protect_latex(text)
    text = escape_latex_special(text)
    text = markdown_to_latex(text)
    text = restore_latex(text, protected)
    return text
```

### Phase 6: Template Population

**Generate main.tex**:
```latex
\\documentclass[{document_class_options}]{{article}}

% Venue-specific packages
{venue_packages}

\\title{{{title}}}

\\author{{
  {author_block}
}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

{sections}

\\bibliographystyle{{{bib_style}}}
\\bibliography{{references}}

\\end{{document}}
```

### Phase 7: PDF Compilation

**Compile with Tectonic** (handles dependencies automatically):
```bash
tectonic main.tex
# Runs: pdflatex → bibtex → pdflatex → pdflatex
```

**Fallback to pdflatex**:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Phase 8: Reproducibility Artifact Checklist

**Generate artifact_checklist.md**:
```markdown
# Reproducibility Artifact Checklist

## Code Availability
- [x] Source code included
- [x] GitHub repository: {repo_url}
- [x] Dockerfile for environment
- [x] README with instructions

## Data
- [x] Dataset sources documented
- [x] Preprocessing scripts included
- [ ] Raw data (if public domain)

## Experiments
- [x] Random seeds specified
- [x] Hyperparameters documented
- [x] Training scripts included
- [x] Evaluation scripts included

## Results
- [x] Main results reproducible
- [x] Ablation studies reproducible
- [x] Statistical tests included

## Documentation
- [x] Installation guide
- [x] Usage examples
- [x] Expected runtime documented
- [x] Hardware requirements specified

## Venue-Specific
- [x] Anonymization (if double-blind)
- [x] Supplementary material prepared
- [x] Ethics statement (if applicable)
```

---

## Output Files

```
paper_[venue]_[timestamp]/
├── main.tex                      # LaTeX source
├── references.bib                # BibTeX citations
├── main.pdf                      # Compiled PDF
├── figures/                      # All figures (copied from evaluation)
│   ├── comparison.png
│   └── ablation.png
├── tables/                       # LaTeX tables
│   ├── results.tex
│   └── ablation.tex
├── supplementary/                # Additional materials
│   ├── supplement.pdf
│   └── code_submission.zip
├── artifact_checklist.md         # Reproducibility checklist
└── verification_warnings.txt     # Citation warnings, if any
```

---

## Advanced Features

### Citation Search Integration

**Semantic Scholar API**:
```python
def verify_citation_semantic_scholar(title, authors):
    """Verify paper exists using Semantic Scholar."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": f"{title} {authors}"}
    response = requests.get(url, params=params)

    if response.json()["total"] > 0:
        paper = response.json()["data"][0]
        return generate_bibtex_from_api(paper)
    return None
```

**arXiv API**:
```python
import arxiv

def verify_citation_arxiv(arxiv_id):
    """Fetch paper metadata from arXiv."""
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    return generate_bibtex_from_arxiv(paper)
```

### Venue-Specific Formatting

**NeurIPS**:
- 9 pages + unlimited references
- Double-blind review (anonymize)
- neurips_2024.sty required

**ACL**:
- 8 pages + unlimited references
- Mandatory ethics statement
- Must include limitations section
- acl2024.sty required

**OSDI**:
- 12 pages + references
- Single-blind (authors visible)
- Mandatory artifact evaluation
- usenix2024.sty required

### Supplementary Material Generation

```bash
/paper-writing --venue neurips \
  --from-artifacts ./ \
  --generate-supplement

# Creates supplement.pdf with:
# - Additional experiments
# - Full algorithm pseudocode
# - Proof sketches
# - Extended related work
```

---

## Writing Quality Standards

A generated paper must read like a real submission to the target venue — not like a technical report, textbook chapter, or user manual. The following standards are non-negotiable.

### Prose Over Lists

Academic papers communicate through argumentative prose. Lists fragment reasoning into disconnected fragments and destroy logical flow.

**Allowed list usage** (exhaustive):
| Context | Format | Example |
|---------|--------|---------|
| Contributions in Introduction | Short `enumerate` (3-4 items, one line each) | "We make three contributions: (1) ..., (2) ..., (3) ..." |
| Pseudocode | `algorithm` environment | `\begin{algorithm}...\end{algorithm}` |
| Table cells | Tabular content | `\begin{tabular}...\end{tabular}` |
| Formal definitions with conditions | `enumerate` for numbered conditions | Definition with conditions (i), (ii), (iii) |

**Forbidden list usage**: Everything else. Specifically:
- Proof steps must be continuous reasoning, not numbered items
- Experimental observations must be interpretive paragraphs, not "Key Observations: 1, 2, 3"
- Discussion points must be developed paragraphs, not bullet summaries
- Future work must be a narrative paragraph, not a wish list
- "Motivation and Significance" must be a paragraph, not an enumeration
- Related work must synthesize papers thematically in paragraphs, not list them

### Logical Flow and Transitions

Each section must connect to the one before it and motivate the one after it. The paper should have a narrative arc:

1. **Introduction** poses a question and creates intellectual tension
2. **Related Work** shows what has been tried and what gap remains
3. **Method** resolves the tension by presenting the approach
4. **Experiments** provides evidence that the approach works
5. **Discussion** reflects on what the evidence means
6. **Conclusion** closes the arc

Between sections and within sections, use transitional sentences: "Having established the theoretical bound, we now turn to empirical validation." Avoid mechanical transitions like "Section 3 describes X."

### Paragraph Construction

Each paragraph must:
- Open with a topic sentence stating the paragraph's point
- Develop the point with evidence, reasoning, or examples (4-8 sentences)
- Close with a sentence that either concludes the point or transitions to the next paragraph

Avoid: single-sentence paragraphs, paragraphs that are just a setup line for a list, paragraphs where every sentence starts with the same word.

### Proof and Theorem Presentation

Present proofs as continuous mathematical reasoning:

```latex
% BAD: Numbered step list
\begin{enumerate}
  \item \textbf{Number of Outputs}: There are $N!$ permutations...
  \item \textbf{Information Content}: Each comparison yields 1 bit...
  \item \textbf{Lower Bound}: Therefore we need $\log_2(N!)$...
\end{enumerate}

% GOOD: Flowing proof
\begin{proof}
Consider the set of all $N!$ permutations of the input. A correct
sorting algorithm must distinguish among all of these, producing
a unique output for each. Since each comparison has two outcomes
($<$ or $>$ for distinct elements), it reveals at most one bit of
information. To distinguish among $N!$ possibilities, we therefore
require at least $\log_2(N!)$ comparisons. Applying Stirling's
approximation, $\log_2(N!) = N\log_2 N - N\log_2 e + O(\log N)$,
which gives the $\Omega(N \log N)$ lower bound.
\end{proof}
```

### Result Interpretation

When presenting experimental results, do not just list what the numbers show. Explain *why* the numbers look the way they do and what they mean for the paper's thesis:

```latex
% BAD: Observation list
\textbf{Key Observations}:
\begin{enumerate}
  \item MergeSort achieves the lower bound...
  \item HeapSort stays within $2\times$...
  \item InsertionSort exceeds the bound...
\end{enumerate}

% GOOD: Interpretive prose
Table~\ref{tab:results} reveals a striking pattern. MergeSort's
comparison count matches the theoretical lower bound to within
rounding error for $N \geq 1000$, providing direct empirical
confirmation that the information-theoretic bound is tight.
HeapSort, while asymptotically optimal, consistently uses roughly
twice the minimum number of comparisons — a consequence of its
heap-maintenance overhead during the extraction phase. The contrast
with InsertionSort is dramatic: its ratio to the lower bound grows
linearly with $N$, reflecting the $O(N^2)$ worst case and
underscoring precisely why the $\Omega(N \log N)$ bound matters
for practical algorithm selection.
```

### Technical Quality

Every generated paper must also satisfy:
- **100% verified citations** (zero hallucinations)
- **Compiles without errors** (LaTeX → PDF)
- **Quantitative results** (numbers from evaluation_report.md)
- **Complexity analysis** (Big-O notation where relevant)
- **Proper figures** (all referenced, high DPI)
- **Error bars** on experimental results
- **Reproducibility info** (code, data, hyperparameters)

---

## Common Pitfalls

### ❌ Writing Style Pitfalls (Most Common)

**1. List-itis: Replacing prose with bullet points**
The single most common failure. Every observation, every discussion point, every proof step gets turned into an `\begin{enumerate}` or `\begin{itemize}`. The result reads like lecture notes, not a paper. If you find yourself writing `\begin{itemize}` outside of the allowed contexts (see Writing Quality Standards), stop and rewrite as a paragraph.

**2. Bold-label-colon pattern as disguised lists**
```latex
% BAD: Disguised list in running text
\textbf{Tightness}: MergeSort achieves the lower bound.
\textbf{Universality}: The bound applies across families.
\textbf{Scalability}: Results hold from 100 to 10,000.

% GOOD: Real paragraph
The experimental results confirm three aspects of the
theoretical bound. First, MergeSort achieves the lower bound
exactly, demonstrating tightness. The bound also proves
universal across algorithm families...
```

**3. Missing transitions between sections**
Sections that start cold without connecting to what came before. Every section opening should acknowledge the preceding discussion and explain why the reader is now moving to this topic.

**4. "Paper Organization" subsections**
A mechanical list of "Section 2 does X, Section 3 does Y" wastes space and bores reviewers. Weave the roadmap into the introduction's final paragraph naturally.

**5. Future Work as a wish list**
```latex
% BAD:
\begin{enumerate}
  \item Parallel sorting bounds
  \item Adaptive sorting
  \item Approximate sorting
  \item Quantum sorting
\end{enumerate}

% GOOD:
The most promising extension concerns parallel sorting, where
the relationship between depth and total work remains poorly
understood. Adaptive sorting also merits investigation: while
Timsort exploits existing order in practice, tight lower bounds
for various measures of presortedness are still open...
```

### ❌ Technical Pitfalls

**6. Hallucinated Citations**
```latex
% BAD: Made-up paper
\\cite{smith2024efficient}  % This paper doesn't exist!

% GOOD: Verified citation
\\cite{vaswani2017attention}  % Transformer paper, verified
```

**7. Markdown in LaTeX**
```latex
% BAD:
**Our method** achieves #1 results

% GOOD:
\\textbf{Our method} achieves \\#1 results
```

**8. Missing Error Bars**
```latex
% BAD:
Our method: 94.2\\%

% GOOD:
Our method: $94.2 \\pm 0.3\\%$
```

**9. No Statistical Tests**
```latex
% BAD:
"significantly better"

% GOOD:
"significantly better ($p < 0.001$, Cohen's $d=1.2$)"
```

---

## Examples

### Example 1: ML Paper
```bash
/paper-writing \
  --venue neurips \
  --from-artifacts ./ \
  --title "Efficient Attention Mechanisms for Long Sequences"

# Output: paper_neurips_20260201/main.pdf
```

### Example 2: Systems Paper
```bash
/paper-writing \
  --venue osdi \
  --from-artifacts ./ \
  --title "ScalableDB: A Distributed Key-Value Store"

# Output: Includes reproducibility artifacts for OSDI requirements
```

### Example 3: NLP Paper
```bash
/paper-writing \
  --venue acl \
  --from-artifacts ./ \
  --ethics-statement ethics.txt \
  --limitations limitations.txt

# Output: ACL format with required sections
```

---

## Integration with Pipeline

**Complete CS research workflow**:
```bash
# Step 1: Literature survey
/literature-survey "efficient attention mechanisms"

# Step 2: Implement method
/method-implementation --from-survey literature_review.md

# Step 3: Evaluate
/experimental-evaluation --workspace workspace/

# Step 4: Write paper (THIS SKILL)
/paper-writing \
  --venue neurips \
  --from-survey literature_review.md \
  --from-implementation workspace/ \
  --from-evaluation experiments/

# Output: Complete NeurIPS submission package
```

---

## References

- [NeurIPS Style Guide](https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles)
- [ACL Author Guidelines](https://www.aclweb.org/portal/content/acl-author-guidelines)
- [OSDI Submission Guide](https://www.usenix.org/conference/osdi24/call-for-papers)
- [LaTeX Best Practices](https://www.overleaf.com/learn)
- [Citation Management](https://www.bibtex.org/)

---

**Skill Version**: 1.0.0
**Last Updated**: 2026-02-01
**Maintainer**: Claude Code Scientific Skills
