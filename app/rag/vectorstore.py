from __future__ import annotations
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "nimbuscart_support_kb"


def build_vector_store(
    chunks: list[Document],
    persist_directory: Path | None = None,
    reset: bool = True,
) -> Chroma:
    """Embed chunks and write them into a persistent Chroma collection.

    Args:
        chunks: Output of `split_documents()`.
        persist_directory: Where Chroma writes its SQLite + index files.
        reset: Delete any existing store first. Default True because re-running
            ingestion without a reset appends DUPLICATE chunks, and duplicates
            crowd out genuinely different results in the top-k. Silent
            duplication is one of the most common RAG bugs.

    Returns:
        A ready-to-query `Chroma` vector store.
    """
    if not chunks:
        raise ValueError("build_vector_store() received no chunks to index.")

    directory = persist_directory or settings.vector_store_path

    if reset and directory.exists():
        logger.info("Removing existing vector store at %s", directory)
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)

    logger.info("Embedding and indexing %d chunks...", len(chunks))

    # `from_documents` does three things in one call:
    #   1. calls embeddings.embed_documents() on every chunk's page_content
    #   2. stores (vector, text, metadata, id) rows in the collection
    #   3. builds the HNSW index used for fast nearest-neighbour search
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(directory),
        # Tell Chroma to rank by cosine distance. The default is L2 (squared
        # euclidean). With normalised vectors the ranking is equivalent, but
        # cosine gives us scores in a range we can reason about.
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.info("Vector store built at %s (%d vectors)", directory, store._collection.count())
    return store


def load_vector_store(persist_directory: Path | None = None) -> Chroma:
    """Open an EXISTING vector store from disk without re-embedding anything.

    This is what the API server calls on startup. Opening is near-instant;
    building takes seconds to minutes. Keeping the two operations in separate
    functions is what makes the server start fast.

    Raises:
        FileNotFoundError: If ingestion has never been run.
    """
    directory = persist_directory or settings.vector_store_path

    if not directory.exists() or not any(directory.iterdir()):
        raise FileNotFoundError(
            f"No vector store found at {directory}.\n"
            "Build it first with:  python -m scripts.ingest"
        )

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = store._collection.count()
    if count == 0:
        raise FileNotFoundError(
            f"Vector store at {directory} exists but is empty. "
            "Re-run:  python -m scripts.ingest"
        )

    logger.info("Loaded vector store from %s (%d vectors)", directory, count)
    return store


def search_with_scores(
    store: Chroma, query: str, k: int | None = None
) -> list[tuple[Document, float]]:
    """Search and return (chunk, SIMILARITY) pairs, highest similarity first.

    Chroma's own method returns cosine DISTANCE where lower is better. We convert
    to similarity (higher is better) so every score in this project reads the
    same way and can be compared against a single threshold.
    """
    top_k = k or settings.retriever_top_k
    results = store.similarity_search_with_score(query, k=top_k)
    # similarity = 1 - distance, clamped to [0, 1] to absorb float noise.
    return [(doc, max(0.0, min(1.0, 1.0 - distance))) for doc, distance in results]


if __name__ == "__main__":
    #     python -m app.rag.vectorstore
    from app.config import setup_logging

    setup_logging()
    vs = load_vector_store()

    for question in [
        "how long does shipping take?",
        "can I return something after 15 days?",
        "my payment was charged twice",
    ]:
        print(f"\nQ: {question}")
        for doc, score in search_with_scores(vs, question, k=3):
            preview = doc.page_content[:90].replace("\n", " ")
            print(f"  {score:.3f}  [{doc.metadata['source']}] {preview}...")
