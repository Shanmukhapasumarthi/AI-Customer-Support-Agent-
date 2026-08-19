from __future__ import annotations
from langchain_groq import ChatGroq
from app.config import settings


def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatGroq:
    """Build the chat model this project uses everywhere.

    Args:
        temperature: Override the configured randomness. 0.0 makes the model
            as deterministic as possible -- correct for customer support,
            where we want the same policy answer every time, not creativity.
        max_tokens: Override the response length cap.

    Returns:
        A configured `ChatGroq` instance implementing LangChain's
        `BaseChatModel` interface.
    """
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
        # Retry transient network / rate-limit failures instead of surfacing
        # a 429 to a customer mid-conversation.
        max_retries=2,
        timeout=30,
    )
