from __future__ import annotations
import logging
import sys
import time

from app.config import setup_logging
from app.rag.loader import load_knowledge_base
from app.rag.splitter import split_documents
from app.rag.vectorstore import build_vector_store, search_with_scores

logger = logging.getLogger(__name__)

# Sanity-check queries run after the build. If any of these returns nothing, the
# index is broken and we want to know NOW, not when a customer asks.
SMOKE_QUERIES = [
    "how long does shipping take",
    "refund window and processing time",
    "can I return an item after 15 days",
    "what does the warranty cover",
    "difference between the X1 and the Mini",
]


def ingest(verify: bool = True) -> int:
    """Run the ingestion pipeline. Returns the number of chunks indexed."""
    started = time.perf_counter()

    logger.info("STEP 1/4  Loading documents...")
    documents = load_knowledge_base()

    logger.info("STEP 2/4  Splitting into chunks...")
    chunks = split_documents(documents)

    logger.info("STEP 3/4  Embedding and indexing (this is the slow part)...")
    store = build_vector_store(chunks, reset=True)

    elapsed = time.perf_counter() - started
    logger.info("Indexed %d chunks in %.1f seconds.", len(chunks), elapsed)

    if verify:
        logger.info("STEP 4/4  Verifying retrieval...")
        failures = 0
        for query in SMOKE_QUERIES:
            results = search_with_scores(store, query, k=2)
            if not results:
                logger.error("  FAIL  %-45s no results", query)
                failures += 1
                continue
            top_doc, top_score = results[0]
            logger.info(
                "  ok    %-45s %.3f  %s",
                query, top_score, top_doc.metadata["source"],
            )
        if failures:
            raise RuntimeError(
                f"{failures} smoke queries returned nothing. The index is not "
                "usable. Check that data/knowledge_base/ contains the policy files."
            )

    return len(chunks)


if __name__ == "__main__":
    setup_logging()
    try:
        count = ingest()
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed: %s: %s", type(exc).__name__, exc)
        sys.exit(1)

    print(f"\nVector store built with {count} chunks.")
    print("Next: python -m scripts.seed_database   (if you have not already)")
    sys.exit(0)
