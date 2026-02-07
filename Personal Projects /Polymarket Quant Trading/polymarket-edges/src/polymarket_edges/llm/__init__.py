"""LLM providers for rules extraction."""

from polymarket_edges.llm.provider import LLMProvider
from polymarket_edges.llm.openai_provider import OpenAIProvider
from polymarket_edges.llm.local_provider import LocalProvider

__all__ = ["LLMProvider", "OpenAIProvider", "LocalProvider"]


def get_provider(provider_type: str = "local") -> LLMProvider:
    """Factory function to get LLM provider.

    Args:
        provider_type: Type of provider ("openai" or "local")

    Returns:
        LLMProvider instance
    """
    if provider_type.lower() == "openai":
        return OpenAIProvider()
    elif provider_type.lower() == "local":
        return LocalProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
