from __future__ import annotations
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings
from app.rag.vectorstore import load_vector_store

logger = logging.getLogger(__name__)

# Chunks scoring below this similarity are treated as irrelevant and dropped.
# Tuned against the evaluation set in scripts/evaluate.py: high enough to reject
# off-topic questions ("who is the CEO?"), low enough to keep genuine paraphrases
# ("when will my package get here?"). Raise it if the agent answers off-topic
# questions; lower it if it says "I don't know" to questions it should handle.
SIMILARITY_THRESHOLD = 0.25


def build_retriever(
    store: Chroma | None = None,
    k: int | None = None,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> VectorStoreRetriever:
    """Create the retriever the RAG chain and the agent both use.

    Args:
        store: An open vector store. Loaded from disk if omitted.
        k: How many chunks to return. More chunks = more context but more noise
            and cost. 4 is a good default for 800-character chunks.
        score_threshold: Minimum cosine similarity to keep a chunk.

    Returns:
        A `VectorStoreRetriever`, which is a Runnable: `.invoke(str)` gives you
        `list[Document]`.
    """
    vector_store = store or load_vector_store()
    top_k = k or settings.retriever_top_k

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            # NOTE: Chroma applies this threshold to the SIMILARITY it computes
            # internally (higher is better), not to the raw distance.
            "score_threshold": score_threshold,
        },
    )

    logger.info("Retriever ready (k=%d, threshold=%.2f)", top_k, score_threshold)
    return retriever


def format_documents(documents: list[Document]) -> str:
    """Turn retrieved Documents into the text block that goes into the prompt.

    THIS FUNCTION IS THE BRIDGE between retrieval and generation, and it is where
    grounding actually happens. The LLM never sees a `Document` object -- it sees
    a string. So the string has to carry the source label with each excerpt,
    otherwise the model cannot cite accurately and will invent a source name.

    Output looks like:

        [Source 1: Refund Policy (refund_policy.md)]
        NimbusCart provides refunds within 30 calendar days...

        [Source 2: FAQ (faq.md)]
        How long do refunds take? ...
    """
    if not documents:
        # An explicit sentinel rather than an empty string. An empty string in a
        # prompt looks like a formatting bug to the model and invites it to fill
        # the gap from memory. A clear statement of absence tells it to say so.
        return "NO RELEVANT DOCUMENTS FOUND IN THE KNOWLEDGE BASE."

    blocks: list[str] = []
    for i, doc in enumerate(documents, start=1):
        title = doc.metadata.get("title", "Knowledge Base")
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Source {i}: {title} ({source})]\n{doc.page_content.strip()}")

    return "\n\n".join(blocks)


def unique_sources(documents: list[Document]) -> list[str]:
    """Distinct source filenames, preserving retrieval order.

    Used to populate the `sources` field of the structured response so the UI can
    show the customer which documents backed the answer.
    """
    seen: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source")
        if source and source not in seen:
            seen.append(source)
    return seen


if __name__ == "__main__":
    #     python -m app.rag.retriever
    from app.config import setup_logging

    setup_logging()
    r = build_retriever()

    for question in [
        "what is your refund policy?",
        "how long does shipping take?",
        "who is the CEO of the company?",  # deliberately unanswerable
    ]:
        docs = r.invoke(question)
        print(f"\nQ: {question}")
        print(f"   retrieved {len(docs)} chunks from {unique_sources(docs) or 'nothing'}")
        if not docs:
            print("   -> empty result. The agent will correctly say it does not know.")
