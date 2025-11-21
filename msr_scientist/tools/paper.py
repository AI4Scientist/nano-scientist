"""
Paper drafting and compilation tools using PyLaTeX.
"""

from smolagents import tool
from typing import Dict, List, Optional
import json
import os
from pathlib import Path


@tool
def draft_research_paper(
    title: str,
    authors: str,
    abstract: str,
    sections: str,
    output_dir: str = "outputs/papers"
) -> str:
    """
    Generate a LaTeX research paper draft and compile it to PDF.

    This tool creates a structured research paper using PyLaTeX and
    compiles it to PDF using pdflatex.

    Args:
        title: Paper title
        authors: Authors (comma-separated or single string)
        abstract: Paper abstract
        sections: JSON string containing sections with titles and content
        output_dir: Directory to save the paper

    Returns:
        A JSON string with paper generation status and file paths
    """
    result = {
        "title": title,
        "status": "draft_created",
        "latex_file": None,
        "pdf_file": None,
        "error": None
    }

    try:
        from pylatex import Document, Section, Subsection, Command
        from pylatex.utils import italic, NoEscape

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Parse sections
        sections_data = json.loads(sections) if isinstance(sections, str) else sections

        # Create document
        doc = Document()

        # Preamble
        doc.preamble.append(Command('title', title))
        doc.preamble.append(Command('author', authors))
        doc.preamble.append(Command('date', NoEscape(r'\today')))

        # Title
        doc.append(NoEscape(r'\maketitle'))

        # Abstract
        with doc.create(Section('Abstract', numbering=False)):
            doc.append(abstract)

        # Add sections
        for section_data in sections_data:
            section_title = section_data.get('title', 'Untitled Section')
            section_content = section_data.get('content', '')
            subsections = section_data.get('subsections', [])

            with doc.create(Section(section_title)):
                if section_content:
                    doc.append(section_content)

                # Add subsections if any
                for subsection_data in subsections:
                    subsection_title = subsection_data.get('title', 'Untitled Subsection')
                    subsection_content = subsection_data.get('content', '')

                    with doc.create(Subsection(subsection_title)):
                        doc.append(subsection_content)

        # Generate file name from title
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:50]
        base_name = f"paper_{safe_title}"

        # Save LaTeX file
        latex_path = Path(output_dir) / base_name
        doc.generate_tex(str(latex_path))
        result["latex_file"] = f"{latex_path}.tex"

        # Try to compile to PDF
        try:
            doc.generate_pdf(str(latex_path), clean_tex=False)
            result["pdf_file"] = f"{latex_path}.pdf"
            result["status"] = "compiled"
        except Exception as pdf_error:
            result["status"] = "latex_only"
            result["error"] = f"PDF compilation failed: {str(pdf_error)}"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


@tool
def add_paper_section(
    paper_path: str,
    section_title: str,
    section_content: str,
    position: int = -1
) -> str:
    """
    Add or update a section in an existing LaTeX paper.

    Args:
        paper_path: Path to the .tex file
        section_title: Title of the section to add/update
        section_content: Content of the section
        position: Position to insert section (-1 for append)

    Returns:
        A JSON string with update status
    """
    result = {
        "paper_path": paper_path,
        "section_title": section_title,
        "status": "updated",
        "error": None
    }

    try:
        # Read existing LaTeX file
        with open(paper_path, 'r') as f:
            content = f.read()

        # Create section LaTeX
        section_latex = f"\\section{{{section_title}}}\n{section_content}\n\n"

        # Find position to insert
        if position == -1:
            # Append before \end{document}
            end_doc_pos = content.rfind(r'\end{document}')
            if end_doc_pos != -1:
                content = content[:end_doc_pos] + section_latex + content[end_doc_pos:]
            else:
                content += section_latex
        else:
            # Insert at specific position (simplified)
            content += section_latex

        # Write back
        with open(paper_path, 'w') as f:
            f.write(content)

        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


@tool
def format_results_table(
    headers: str,
    rows: str,
    caption: str = ""
) -> str:
    """
    Generate a LaTeX table from experimental results.

    Args:
        headers: JSON list of column headers
        rows: JSON list of rows (each row is a list of values)
        caption: Table caption

    Returns:
        LaTeX table code as a string
    """
    try:
        from pylatex import Table, Tabular
        from pylatex.utils import NoEscape

        headers_list = json.loads(headers) if isinstance(headers, str) else headers
        rows_list = json.loads(rows) if isinstance(rows, str) else rows

        # Create table
        table = Table(position='h!')
        if caption:
            table.add_caption(caption)

        # Create tabular
        col_spec = 'l' * len(headers_list)
        with table.create(Tabular(col_spec)) as tabular:
            tabular.add_hline()
            tabular.add_row(headers_list)
            tabular.add_hline()
            for row in rows_list:
                tabular.add_row(row)
            tabular.add_hline()

        return table.dumps()

    except Exception as e:
        return f"% Error generating table: {str(e)}"


class PaperDrafter:
    """
    Manages research paper drafting and compilation.
    """

    def __init__(self, output_dir: str = "outputs/papers"):
        self.output_dir = output_dir
        self.papers = []
        os.makedirs(output_dir, exist_ok=True)

    def draft(
        self,
        title: str,
        authors: str,
        abstract: str,
        sections: List[Dict]
    ) -> Dict:
        """Draft a research paper."""
        sections_json = json.dumps(sections)
        result_json = draft_research_paper(
            title, authors, abstract, sections_json, self.output_dir
        )
        result = json.loads(result_json)
        self.papers.append(result)
        return result

    def add_section(self, paper_path: str, title: str, content: str) -> Dict:
        """Add a section to an existing paper."""
        result_json = add_paper_section(paper_path, title, content)
        return json.loads(result_json)

    def create_table(self, headers: List[str], rows: List[List], caption: str = "") -> str:
        """Create a LaTeX table."""
        return format_results_table(
            json.dumps(headers),
            json.dumps(rows),
            caption
        )

    def list_papers(self):
        """List all drafted papers."""
        return self.papers
