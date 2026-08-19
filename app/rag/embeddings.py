from __future__ import annotations
import logging
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model.

    `@lru_cache` matters a lot here. Loading a sentence-transformer means
    reading ~90 MB of weights and building a PyTorch model -- roughly 2-5
    seconds. Without caching, every FastAPI request would reload it. With
    caching, it loads once per process and every later call is instant.

    Returns:
        A LangChain `Embeddings` object exposing two methods:
          * `embed_documents(list[str]) -> list[list[float]]`  (ingestion)
          * `embed_query(str) -> list[float]`                  (search)

        Some models prepend different instructions for those two cases, which is
        why the interface separates them.
    """
    logger.info("Loading embedding model: %s (first call downloads ~90 MB)",
                settings.embedding_model)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        # Force CPU: this model is small and CPU inference avoids CUDA setup
        # issues entirely. Change to "cuda" if you have a GPU and want speed.
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            # Normalising to unit length means cosine similarity and dot product
            # give the same ranking, and similarity scores land in a predictable
            # 0-1 range. This makes the confidence numbers in Phase 14 stable.
            "normalize_embeddings": True,
        },
    )

    logger.info("Embedding model ready.")
    return embeddings


def embedding_dimension() -> int:
    """Return the vector length this model produces (384 for all-MiniLM-L6-v2).

    Handy sanity check: if you swap the model in .env and this number changes,
    your existing vector store is incompatible and MUST be rebuilt.
    """
    return len(get_embeddings().embed_query("dimension probe"))


if __name__ == "__main__":
    #     python -m app.rag.embeddings
    # Demonstrates that similar MEANING produces a high similarity score even
    # when the two sentences share almost no words.
    from app.config import setup_logging

    setup_logging()
    emb = get_embeddings()

    print(f"Model     : {settings.embedding_model}")
    print(f"Dimensions: {embedding_dimension()}")

    def cosine(a: list[float], b: list[float]) -> float:
        # Vectors are already normalised, so cosine similarity is just the
        # dot product. No division needed.
        return sum(x * y for x, y in zip(a, b))

    query = "when will my package get here?"
    candidates = [
        "Standard Shipping takes 5 to 7 business days after dispatch.",
        "Refunds are processed within 5-7 business days after inspection.",
        "The BrewMaster Duo has a 15 bar pump and a 1.8 litre tank.",
        "Our office cat is named Biscuit.",
    ]

    qv = emb.embed_query(query)
    print(f"\nQuery: {query!r}\n")
    scored = [(cosine(qv, emb.embed_query(c)), c) for c in candidates]
    for score, text in sorted(scored, reverse=True):
        print(f"  {score:.3f}  {text}")
    print("\nNote: the shipping sentence wins despite sharing zero keywords "
          "with the query. That is semantic search.")
