"""config.py - Centralized model configuration using LiteLLM

Loads environment variables and provides unified model configuration
for all pipeline stages. Uses LiteLLM for universal model support.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def normalize_model_id(self) -> str:
        """Ensure model_id has provider prefix.

        Returns:
            Normalized model ID with provider prefix
        """
        if '/' in self.model_id:
            return self.model_id

        # Auto-detect provider from common patterns
        if self.model_id.startswith('gpt-'):
            return f'openai/{self.model_id}'
        elif self.model_id.startswith('claude-'):
            return f'anthropic/{self.model_id}'
        elif 'gemini' in self.model_id.lower():
            return f'vertex_ai/{self.model_id}'
        elif any(x in self.model_id.lower() for x in ['llama', 'mistral', 'mixtral']):
            # Common HuggingFace models
            return self.model_id  # Keep as-is, may need HF hub format

        # Default to OpenAI for backward compatibility with generic names
        return f'openai/{self.model_id}'


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    stage1_model: ModelConfig  # Research & Planning
    stage2_model: ModelConfig  # Execute & Analyze
    stage3_model: ModelConfig  # Report Generation
    agent_model: ModelConfig   # Main orchestrator

    # Stage-specific settings
    stage1_max_loops: int = 5
    stage1_search_api: str = 'tavily'
    stage2_step_limit: int = 50
    stage2_cost_limit: float = 2.0

    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """Load configuration from environment variables.

        Returns:
            PipelineConfig instance loaded from environment
        """
        # Load model configurations
        stage1_model = ModelConfig(
            model_id=os.getenv('STAGE1_MODEL', 'anthropic/claude-haiku-4-5-20251001'),
            temperature=float(os.getenv('STAGE1_TEMPERATURE', '0.0'))
        )

        stage2_model = ModelConfig(
            model_id=os.getenv('STAGE2_MODEL', 'anthropic/claude-haiku-4-5-20251001'),
            temperature=float(os.getenv('STAGE2_TEMPERATURE', '0.0'))
        )

        stage3_model = ModelConfig(
            model_id=os.getenv('STAGE3_MODEL', 'anthropic/claude-haiku-4-5-20251001'),
            temperature=float(os.getenv('STAGE3_TEMPERATURE', '0.0'))
        )

        agent_model = ModelConfig(
            model_id=os.getenv('AGENT_MODEL', 'anthropic/claude-haiku-4-5-20251001'),
            temperature=float(os.getenv('AGENT_TEMPERATURE', '0.0'))
        )

        return cls(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            stage3_model=stage3_model,
            agent_model=agent_model,
            stage1_max_loops=int(os.getenv('STAGE1_MAX_LOOPS', '5')),
            stage1_search_api=os.getenv('STAGE1_SEARCH_API', 'tavily'),
            stage2_step_limit=int(os.getenv('STAGE2_STEP_LIMIT', '50')),
            stage2_cost_limit=float(os.getenv('STAGE2_COST_LIMIT', '2.0'))
        )

    def validate(self) -> None:
        """Validate configuration and check required API keys.

        Raises:
            ValueError: If required API keys are missing or configuration is invalid
        """
        errors = []

        # Check that models are properly formatted and have required API keys
        models = [
            ('AGENT_MODEL', self.agent_model),
            ('STAGE1_MODEL', self.stage1_model),
            ('STAGE2_MODEL', self.stage2_model),
            ('STAGE3_MODEL', self.stage3_model)
        ]

        for name, model_config in models:
            normalized = model_config.normalize_model_id()
            provider = normalized.split('/')[0] if '/' in normalized else 'unknown'

            # Check for required API keys based on provider
            if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
                errors.append(f'{name} uses OpenAI ({model_config.model_id}) but OPENAI_API_KEY not set')
            elif provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
                errors.append(f'{name} uses Anthropic ({model_config.model_id}) but ANTHROPIC_API_KEY not set')
            elif provider in ['huggingface', 'hf'] and not os.getenv('HF_TOKEN'):
                errors.append(f'{name} uses HuggingFace ({model_config.model_id}) but HF_TOKEN not set')

        # Check search API key
        if self.stage1_search_api == 'tavily' and not os.getenv('TAVILY_API_KEY'):
            errors.append('STAGE1_SEARCH_API is tavily but TAVILY_API_KEY not set')
        elif self.stage1_search_api == 'perplexity' and not os.getenv('PERPLEXITY_API_KEY'):
            errors.append('STAGE1_SEARCH_API is perplexity but PERPLEXITY_API_KEY not set')

        if errors:
            raise ValueError(
                'Configuration validation failed:\n' +
                '\n'.join(f'  - {e}' for e in errors)
            )


# Global config instance (singleton pattern)
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get the global configuration instance.

    Loads configuration from environment variables on first call,
    then caches and returns the same instance on subsequent calls.

    Returns:
        PipelineConfig instance

    Raises:
        ValueError: If configuration validation fails
    """
    global _config
    if _config is None:
        _config = PipelineConfig.from_env()
        _config.validate()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance.

    Useful for testing or when environment variables change.
    """
    global _config
    _config = None
