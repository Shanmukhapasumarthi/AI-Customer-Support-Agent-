from __future__ import annotations
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Construct the text splitter used for the whole knowledge base."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        # Measured in characters, not tokens. Simple, dependency-free, and
        # accurate enough: roughly 4 characters per token for English.
        length_function=len,
        # Markdown-aware separator order. We put markdown headings FIRST so a
        # chunk boundary prefers to land at a section break. This keeps
        # "## Refund processing time" together with the text underneath it.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        # Records the character offset of each chunk in the parent document.
        # Useful for debugging "where exactly did this chunk come from?".
        add_start_index=True,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split whole documents into retrieval-sized chunks.

    Each chunk keeps its parent's metadata (source, title, policy_area) and gains
    a `chunk_index` so we can talk about "chunk 3 of refund_policy.md".

    Args:
        documents: Whole documents from `load_knowledge_base()`.

    Returns:
        A flat list of chunk Documents, ready to be embedded.
    """
    if not documents:
        raise ValueError("split_documents() received an empty document list.")

    splitter = build_splitter()

    # `split_documents` (plural) preserves metadata automatically. The
    # lower-level `split_text` does NOT -- it returns bare strings and you lose
    # every citation. Always use the Document-level method.
    chunks = splitter.split_documents(documents)

    # Number chunks within each source file, for readable debugging output.
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        index = per_source_counter.get(source, 0)
        chunk.metadata["chunk_index"] = index
        per_source_counter[source] = index + 1

    sizes = [len(c.page_content) for c in chunks]
    logger.info(
        "Split %d documents into %d chunks (size=%d, overlap=%d). "
        "Chunk chars: min=%d avg=%d max=%d",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
        min(sizes),
        sum(sizes) // len(sizes),
        max(sizes),
    )
    return chunks


if __name__ == "__main__":
    #     python -m app.rag.splitter
    from app.config import setup_logging
    from app.rag.loader import load_knowledge_base

    setup_logging()
    docs = load_knowledge_base()
    chunks = split_documents(docs)

    print("\nChunks per source file:")
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.metadata["source"]] = counts.get(c.metadata["source"], 0) + 1
    for source, n in sorted(counts.items()):
        print(f"  {source:<28} {n:>3} chunks")

    print(f"\n--- Example chunk (chunk 1 of {chunks[1].metadata['source']}) ---")
    print(f"metadata: {chunks[1].metadata}")
    print(chunks[1].page_content[:400])
