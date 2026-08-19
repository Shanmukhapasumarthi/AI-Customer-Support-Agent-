"""
PHASE 12 - CONNECTING RAG TO THE AGENT

THE KEY IDEA: RETRIEVAL BECOMES A TOOL.

In Phase 8 the RAG chain was the whole application: every question went through
retrieval whether it needed to or not. That is wrong for a support agent, because
"where is ORD1001?" has no useful answer in the policy documents.

The fix is to demote retrieval from "the pipeline" to "one capability among
several". We wrap the retriever in a tool, hand it to the agent alongside the
order and product tools, and let the LLM decide.

That single change is what turns a CHAIN into an AGENT:

    CHAIN:  question -> retrieve -> LLM -> answer          (path is fixed)
    AGENT:  question -> LLM picks a tool -> observe -> ... -> answer
                                                          (path is chosen at runtime)

The agent can also do things a chain cannot:
  * call NO tool at all ("hello" needs no lookup),
  * call the SAME tool twice with different queries,
  * call TWO DIFFERENT tools and combine the results -- which is what makes
    "can I still return ORD1006?" work: it needs the delivery date from the
    database AND the return window from the knowledge base.

LangChain ships `create_retriever_tool` for exactly this, and we use it, but we
also expose a hand-written variant so you can see there is no magic: a retriever
tool is just a function that calls `retriever.invoke()` and formats the result.
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field

from app.rag.retriever import build_retriever, format_documents

logger = logging.getLogger(__name__)

# Module-level cache. The retriever holds the embedding model and the open
# Chroma collection, both expensive to construct. Build once per process.
_retriever: VectorStoreRetriever | None = None

# Tracks which knowledge-base files the most recent search touched. The agent's
# final structured response reports its sources, but the LLM only ever sees the
# tool's TEXT output -- asking it to remember and repeat filenames accurately is
# exactly the kind of thing models get subtly wrong. So we record the real
# sources here, in Python, and read them back after the run. Never trust the
# model to report its own provenance.
_last_sources: list[str] = []


class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        description="A natural-language search query describing the policy "
                    "information needed, for example 'refund window and "
                    "processing time' or 'can items be returned after 15 days'. "
                    "Write it as a descriptive phrase, not as a single keyword."
    )


def get_retriever() -> VectorStoreRetriever:
    """Return the shared retriever, building it on first use."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached retriever so the next call re-reads the vector store.

    Used by the POST /knowledge/reload endpoint after re-ingestion, so updated
    policy documents take effect without restarting the API server.
    """
    global _retriever
    _retriever = None
    logger.info("Retriever cache cleared; will reload on next use.")


def get_last_sources() -> list[str]:
    """Source filenames from the most recent knowledge-base search."""
    return list(_last_sources)


def clear_last_sources() -> None:
    """Reset source tracking. Called at the start of every agent run so sources
    from a previous customer's question cannot leak into this one's citations."""
    _last_sources.clear()


def _search_knowledge_base(query: str) -> str:
    """Implementation behind search_knowledge_base."""
    documents = get_retriever().invoke(query)

    if not documents:
        logger.info("Knowledge base search found nothing for %r", query)
        return (
            "NO RELEVANT DOCUMENTS FOUND. The knowledge base does not cover "
            "this topic. Tell the customer you do not have that information "
            "rather than answering from general knowledge."
        )

    for doc in documents:
        source = doc.metadata.get("source")
        if source and source not in _last_sources:
            _last_sources.append(source)

    logger.info(
        "Knowledge base search %r returned %d chunks from %s",
        query, len(documents), _last_sources,
    )
    return format_documents(documents)


search_knowledge_base_tool = StructuredTool.from_function(
    func=_search_knowledge_base,
    name="search_knowledge_base",
    description=(
        "Search NimbusCart's official policy documents and FAQ. Use this for "
        "ANY question about company policy: refunds, returns, exchanges, "
        "shipping times and costs, cancellations, warranty coverage, payment "
        "methods, product specifications and comparisons, memberships, or "
        "support hours. This is the ONLY approved source for policy "
        "information -- never answer a policy question from your own general "
        "knowledge about how online stores usually work. Returns excerpts from "
        "the documents with their source filenames."
    ),
    args_schema=KnowledgeSearchInput,
    handle_tool_error=True,
)

KNOWLEDGE_TOOLS = [search_knowledge_base_tool]
