"""Core agent with linear message history, inspired by mini-swe-agent."""

import os
from typing import List, Dict, Any, Optional


class ResearchAgent:
    """Minimalist agent with linear history and LLM integration."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize agent with model configuration.

        Args:
            model: LLM model name (gpt-4, claude-3-5-sonnet-20241022, etc.)
            api_key: API key for the model provider
        """
        self.model = model
        if api_key:
            self.api_key = api_key
        elif "claude" in self.model.lower():
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the research agent."""
        return """You are a research scientist agent with the following capabilities:

1. **Hypothesis Generation**: Analyze research tasks and formulate testable hypotheses
2. **Experimental Code**: Write Python code to test hypotheses using Linux tools and PyPI libraries
3. **Tool Creation**: Generate new Python tools on-the-fly when needed
4. **Paper Writing**: Generate LaTeX research papers documenting findings

You work iteratively:
- Break down research tasks into hypotheses
- Design and run experiments
- Analyze results
- Generate academic papers

When you need a tool that doesn't exist, generate it immediately.
Use bash commands and Python code to conduct experiments.
Be scientific, rigorous, and minimalist in your approach."""

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with messages and return response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            LLM response text
        """
        if "claude" in self.model.lower():
            return self._call_anthropic(messages)
        else:
            return self._call_openai(messages)

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}] + messages
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=messages
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def step(self, user_message: str) -> str:
        """Execute one step of the agent loop.

        Args:
            user_message: User input or observation

        Returns:
            Agent response
        """
        # Linear history: just append message
        self.messages.append({"role": "user", "content": user_message})

        # Get LLM response
        response = self.call_llm(self.messages)

        # Append assistant response to history
        self.messages.append({"role": "assistant", "content": response})

        return response

    def run(self, task: str, max_steps: int = 10) -> List[Dict[str, str]]:
        """Run the agent on a research task.

        Args:
            task: Research task description
            max_steps: Maximum number of interaction steps

        Returns:
            Full message history
        """
        print(f"Starting research task: {task}\n")

        response = self.step(task)
        print(f"Agent: {response}\n")

        for i in range(max_steps - 1):
            # In a full implementation, this would execute commands and return observations
            # For now, we'll let the workflow orchestrator handle this
            break

        return self.messages

    def reset(self):
        """Reset agent state."""
        self.messages = []
