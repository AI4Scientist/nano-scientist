"""research_plan.py - Stage 1: Research & Planning module

Refactored from deep_research.py to provide minimalist research functionality.
Removes Rich console UI and CLI, returns structured data for pipeline integration.
"""

import os
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List

import requests
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()


class SearchAPI(str, Enum):
    """The search API to use for the research assistant."""
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


class Message:
    """Simple message class for LiteLLM compatibility."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class Configuration(BaseModel):
    """The configurable fields for the research assistant."""
    max_web_research_loops: int = 3
    model_id: str = "anthropic/claude-haiku-4-5-20251001"
    search_api: SearchAPI = SearchAPI.TAVILY
    temperature: float = 0.7


# --- Prompts ---
query_writer_instructions = """As a research query generator, create a targeted web search query to gather information about:

{research_topic}

Your goal is to formulate a precise query that will yield relevant, high-quality information about this topic.

Example output:
{{
    "query": "machine learning transformer architecture explained",
    "aspect": "technical architecture",
    "rationale": "Understanding the fundamental structure of transformer models"
}}

Generate your query now:"""

summarizer_instructions = """You are tasked with generating a high-quality, concise summary of web search results related to the user's topic.

For a new summary:
• Structure the main points as a bullet list
• Extract and highlight the most relevant information
• Maintain a logical flow of ideas
• Focus on accuracy and clarity

When extending an existing summary:
• Keep or enhance the bullet-point structure
• Integrate new information seamlessly with existing content
• Add new, relevant details while maintaining coherence
• Skip redundant or irrelevant information
• Ensure the final summary shows clear progression

Format your summary with:
• An introductory paragraph
• Key points as bullet items, only the most important points
• Each bullet should be a complete, informative statement

Begin your summary now:"""

reflection_instructions = """As a research analyst examining our current knowledge about {research_topic}, identify gaps in our understanding and propose targeted follow-up questions.

Focus on:
• Technical details that need clarification
• Implementation specifics that are unclear
• Emerging trends or developments not yet covered
• Practical applications or implications not discussed

Your follow-up query must:
• Be self-contained and include the topic/subject explicitly
• Provide enough context to stand alone as a search query
• Be specific enough to yield relevant results

Example output:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are the performance benchmarks and system requirements for Stalker 2: Heart of Chornobyl?"
}}

Analyze the current summary and provide your insights:"""


class SearchQuery(BaseModel):
    """Model for the search query generation output."""
    query: str = Field(description="The actual search query string")
    aspect: str = Field(description="The specific aspect of the topic being researched")
    rationale: str = Field(
        description="Brief explanation of why this query is relevant"
    )


class ReflectionOutput(BaseModel):
    """Model for the reflection output."""
    knowledge_gap: str = Field(
        description="Description of what information is missing or needs clarification"
    )
    follow_up_query: str = Field(
        description="Specific question to address the identified knowledge gap"
    )


class DeepResearcher:
    """The main class for the deep researcher."""

    def __init__(self, config: Configuration):
        self.config = config

    def get_llm(self, structured_output=None, streaming: bool = False):
        """Get the LLM client using LiteLLM."""
        from config import ModelConfig

        # Normalize model_id
        model_config = ModelConfig(model_id=self.config.model_id)
        normalized_model = model_config.normalize_model_id()

        if structured_output:
            # Return callable for structured completion
            return lambda messages: self._structured_completion(
                normalized_model, messages, structured_output, self.config.temperature
            )
        else:
            # Return callable for text completion
            return lambda messages: self._text_completion(
                normalized_model, messages, self.config.temperature, streaming
            )

    def _format_messages(self, messages) -> list[dict]:
        """Convert message objects to LiteLLM dict format."""
        formatted = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                formatted.append({
                    'role': msg.role,
                    'content': msg.content
                })
            else:
                # Fallback for any other format
                formatted.append({
                    'role': 'user',
                    'content': str(msg)
                })
        return formatted

    def _structured_completion(self, model_id, messages, output_schema, temperature):
        """Make a structured completion call with JSON mode."""
        formatted_messages = self._format_messages(messages)

        # Add instruction for JSON output
        if formatted_messages and formatted_messages[-1]['role'] == 'user':
            formatted_messages[-1]['content'] += "\n\nReturn ONLY valid JSON matching the schema."

        response = litellm.completion(
            model=model_id,
            messages=formatted_messages,
            response_format={"type": "json_object"},
            temperature=temperature
        )

        import json
        content = response.choices[0].message.content

        # Parse and validate against Pydantic schema
        try:
            data = json.loads(content)
            return output_schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            # Try to extract JSON if wrapped in markdown or text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return output_schema(**data)
            raise ValueError(f"Failed to parse structured output: {e}")

    def _text_completion(self, model_id, messages, temperature, streaming):
        """Make a text completion call."""
        formatted_messages = self._format_messages(messages)

        response = litellm.completion(
            model=model_id,
            messages=formatted_messages,
            temperature=temperature,
            stream=streaming
        )

        # Return response object with content attribute for compatibility
        class Response:
            def __init__(self, content):
                self.content = content

        return Response(response.choices[0].message.content)

    def generate_query(self, research_topic: str) -> str:
        """Generate a query for web search."""
        instructions = query_writer_instructions.format(research_topic=research_topic)
        llm_client = self.get_llm(structured_output=SearchQuery)
        result = llm_client(
            [
                Message('system', instructions),
                Message('user', "Generate a query for web search:"),
            ]
        )
        return result.query

    def web_research(self, query: str, loop_count: int):
        """Perform web research on a query."""
        if self.config.search_api == SearchAPI.TAVILY:
            search_results = self.tavily_search(
                query, include_raw_content=True, max_results=1
            )
            search_summary = self.deduplicate_and_format_sources(
                search_results, max_tokens_per_source=2000, include_raw_content=True
            )
        elif self.config.search_api == SearchAPI.PERPLEXITY:
            search_results = self.perplexity_search(query, loop_count)
            search_summary = self.deduplicate_and_format_sources(
                search_results, max_tokens_per_source=2000, include_raw_content=False
            )
        else:
            raise ValueError(f"Unsupported search API: {self.config.search_api}")
        sources = [self.format_sources(search_results)]
        return search_summary, sources

    def summarize_sources(
        self, research_topic: str, current_summary: Optional[str], search_summary: str
    ) -> str:
        """Summarize the sources."""
        if current_summary:
            message = (
                f"<User Input> \n {research_topic} \n <User Input>\n\n"
                f"<Existing Summary> \n {current_summary} \n <Existing Summary>\n\n"
                f"<New Search Results> \n {search_summary} \n <New Search Results>"
            )
        else:
            message = (
                f"<User Input> \n {research_topic} \n <User Input>\n\n"
                f"<Search Results> \n {search_summary} \n <Search Results>"
            )
        llm_client = self.get_llm()
        result = llm_client(
            [
                Message('system', summarizer_instructions),
                Message('user', message),
            ]
        )
        return result.content

    def reflect_on_summary(self, research_topic: str, current_summary: str) -> str:
        """Reflect on the summary."""
        llm_client = self.get_llm(structured_output=ReflectionOutput, streaming=True)
        message = f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {current_summary}"
        result = llm_client(
            [
                Message(
                    'system',
                    reflection_instructions.format(research_topic=research_topic)
                ),
                Message('user', message),
            ]
        )
        return result.follow_up_query

    def finalize_summary(self, running_summary: str, sources_gathered: list) -> str:
        """Finalize the summary with sources."""
        flat_sources = [item for sublist in sources_gathered for item in sublist]
        all_sources = "\n".join(flat_sources)
        final = f"## Summary\n\n{running_summary}\n\n## Sources\n\n{all_sources}"
        return final

    def run_research(self, topic: str) -> Tuple[Dict, List[Dict]]:
        """Run the research and return structured data.

        Returns:
            Tuple of (plan_dict, citations_list)
            - plan_dict: Research plan with hypotheses, methodology, metrics
            - citations_list: List of citation dictionaries
        """
        research_loop_count = 0
        sources_gathered = []
        running_summary = None

        search_query = self.generate_query(topic)

        while research_loop_count < self.config.max_web_research_loops:
            search_summary, sources = self.web_research(search_query, research_loop_count)
            research_loop_count += 1
            sources_gathered.append(sources)
            running_summary = self.summarize_sources(
                topic, running_summary, search_summary
            )
            search_query = self.reflect_on_summary(topic, running_summary)

        final_summary = self.finalize_summary(running_summary, sources_gathered)

        # Extract structured plan from summary using LLM
        plan_dict = self._extract_plan_from_summary(topic, final_summary)

        # Extract citations from sources
        citations_list = self._extract_citations(sources_gathered)

        return plan_dict, citations_list

    def _extract_plan_from_summary(self, topic: str, summary: str) -> Dict:
        """Extract structured research plan from summary using LLM."""
        llm_client = self.get_llm()
        prompt = f"""Based on this research summary, create a structured research plan.

Research Topic: {topic}

Research Summary:
{summary}

Generate a JSON research plan with this structure:
{{
    "task": "Original research task",
    "hypotheses": ["Hypothesis 1", "Hypothesis 2"],
    "methodology": {{
        "approach": "Description of experimental approach",
        "steps": ["Step 1", "Step 2", "Step 3"],
        "tools_needed": ["package1", "package2"]
    }},
    "metrics": {{
        "primary": "Main success metric",
        "secondary": ["Secondary metric 1", "Secondary metric 2"]
    }},
    "expected_deliverables": ["Deliverable 1", "Deliverable 2"]
}}

Return ONLY the JSON, no additional text."""

        result = llm_client([Message('user', prompt)])
        import json
        try:
            plan = json.loads(result.content)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return pure JSON
            import re
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # Ultimate fallback
                plan = {
                    "task": topic,
                    "hypotheses": ["Research and validate findings"],
                    "methodology": {
                        "approach": "Literature review and analysis",
                        "steps": ["Gather information", "Analyze findings", "Document results"],
                        "tools_needed": []
                    },
                    "metrics": {
                        "primary": "Completeness of research",
                        "secondary": []
                    },
                    "expected_deliverables": ["Research report"]
                }
        return plan

    def _extract_citations(self, sources_gathered: List) -> List[Dict]:
        """Extract citation metadata from sources with enhanced BibTeX-compatible fields."""
        citations = []
        citation_id = 1

        flat_sources = [item for sublist in sources_gathered for item in sublist]

        for source_text in flat_sources:
            # Parse markdown links from source text
            import re
            # Pattern: * **Title**\n\n    <url>
            matches = re.findall(r'\*\s+\*\*(.+?)\*\*\s*\n\s*<(.+?)>', source_text)
            for title, url in matches:
                # Use LLM to extract enhanced citation metadata
                enhanced_metadata = self._extract_citation_metadata(title.strip(), url.strip())

                citations.append({
                    "id": citation_id,
                    "title": enhanced_metadata.get("title", title.strip()),
                    "authors": enhanced_metadata.get("authors", "Unknown"),
                    "year": enhanced_metadata.get("year", "n.d."),
                    "url": url.strip(),
                    "publisher": enhanced_metadata.get("publisher", ""),
                    "journal": enhanced_metadata.get("journal", ""),
                    "type": enhanced_metadata.get("type", "misc"),  # article, book, misc, etc.
                    "bibtex_key": f"ref{citation_id}"
                })
                citation_id += 1

        return citations

    def _extract_citation_metadata(self, title: str, url: str) -> Dict:
        """Use LLM to extract enhanced citation metadata for BibTeX."""
        llm_client = self.get_llm()
        prompt = f"""Extract citation metadata from this source for BibTeX format:

Title: {title}
URL: {url}

Based on the title and URL, infer the following fields:
- authors: Author name(s) in "Last, First" format (or "Unknown" if unclear)
- year: Publication year (or "n.d." if not available)
- publisher: Publisher or website name
- journal: Journal name if it's an academic article
- type: BibTeX entry type (article, book, inproceedings, misc, online)

Return a JSON object with these fields. Make reasonable inferences from the URL domain and title.

Example:
{{
    "title": "Machine Learning Basics",
    "authors": "Smith, John and Doe, Jane",
    "year": "2024",
    "publisher": "Nature",
    "journal": "Nature Machine Intelligence",
    "type": "article"
}}

Return ONLY valid JSON."""

        try:
            result = llm_client([Message('user', prompt)])
            import json
            metadata = json.loads(result.content)
        except Exception:
            # Fallback to basic inference from URL
            metadata = self._infer_metadata_from_url(title, url)

        return metadata

    def _infer_metadata_from_url(self, title: str, url: str) -> Dict:
        """Infer basic metadata from URL patterns."""
        import re
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc

        # Try to extract year from URL or title
        year_match = re.search(r'20\d{2}', url + title)
        year = year_match.group(0) if year_match else "n.d."

        # Infer type and publisher from domain
        metadata = {
            "title": title,
            "authors": "Unknown",
            "year": year,
            "publisher": domain,
            "journal": "",
            "type": "online"
        }

        # Domain-specific inference
        if 'arxiv.org' in domain:
            metadata["type"] = "article"
            metadata["publisher"] = "arXiv"
        elif any(x in domain for x in ['nature.com', 'science.org', 'ieee.org', 'acm.org']):
            metadata["type"] = "article"
            metadata["publisher"] = domain.replace('.com', '').replace('.org', '').title()
        elif 'wikipedia.org' in domain:
            metadata["type"] = "misc"
            metadata["publisher"] = "Wikipedia"
        elif any(x in domain for x in ['github.com', 'gitlab.com']):
            metadata["type"] = "misc"
            metadata["publisher"] = "GitHub" if 'github' in domain else "GitLab"

        return metadata

    @staticmethod
    def deduplicate_and_format_sources(
        search_response, max_tokens_per_source, include_raw_content=False
    ):
        """Deduplicate and format the sources (for the detailed summary)."""
        if isinstance(search_response, dict):
            sources_list = search_response["results"]
        elif isinstance(search_response, list):
            sources_list = []
            for response in search_response:
                if isinstance(response, dict) and "results" in response:
                    sources_list.extend(response["results"])
                else:
                    sources_list.extend(response)
        else:
            raise ValueError(
                "Input must be either a dict with 'results' or a list of search results"
            )

        unique_sources = {}
        for source in sources_list:
            if source["url"] not in unique_sources:
                unique_sources[source["url"]] = source

        formatted_text = ""
        for source in unique_sources.values():
            formatted_text += f"Source: {source['title']}\n"
            formatted_text += f"URL: {source['url']}\n"
            formatted_text += (
                f"Content: {source['content']}\n"
            )
            if include_raw_content:
                char_limit = max_tokens_per_source * 4
                raw_content = source.get("raw_content", "")
                if raw_content:
                    if len(raw_content) > char_limit:
                        raw_content = raw_content[:char_limit] + "... [truncated]"
                    formatted_text += f"Full content (truncated to {max_tokens_per_source} tokens): {raw_content}\n\n"
            formatted_text += "---\n"
        return formatted_text.strip()

    @staticmethod
    def format_sources(search_results):
        """Format sources for Markdown output (used in each loop)."""
        formatted_sources = []
        for source in search_results["results"]:
            formatted_sources.append(f"* **{source['title']}**\n\n    <{source['url']}>")
        return "\n".join(formatted_sources)

    @staticmethod
    def tavily_search(query, include_raw_content=True, max_results=3):
        """Search with Tavily."""
        tavily_client = TavilyClient()
        return tavily_client.search(
            query, max_results=max_results, include_raw_content=include_raw_content
        )

    @staticmethod
    def perplexity_search(
        query: str, perplexity_search_loop_count: int
    ) -> Dict[str, Any]:
        """Search with Perplexity."""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources.",
                },
                {"role": "user", "content": query},
            ],
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        results = [
            {
                "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
                "url": citations[0],
                "content": content,
                "raw_content": content,
            }
        ]
        for i, citation in enumerate(citations[1:], start=2):
            results.append(
                {
                    "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
                    "url": citation,
                    "content": "See above for full content",
                    "raw_content": None,
                }
            )
        return {"results": results}
