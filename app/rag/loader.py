from __future__ import annotations
import logging
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from app.config import settings

logger = logging.getLogger(__name__)

# Human-readable titles and a coarse topic label for each knowledge-base file.
# We attach these as metadata so the agent can cite a friendly name instead of a
# filename, and so we could filter by topic later.
DOCUMENT_CATALOG: dict[str, dict[str, str]] = {
    "refund_policy.md": {"title": "Refund Policy", "policy_area": "refund"},
    "return_policy.md": {"title": "Return Policy", "policy_area": "return"},
    "shipping_policy.md": {"title": "Shipping Policy", "policy_area": "shipping"},
    "cancellation_policy.md": {"title": "Cancellation Policy", "policy_area": "cancellation"},
    "warranty_policy.md": {"title": "Warranty Policy", "policy_area": "warranty"},
    "payment_policy.md": {"title": "Payment Policy", "policy_area": "payment"},
    "product_information.md": {"title": "Product Information", "policy_area": "product"},
    "faq.md": {"title": "Frequently Asked Questions", "policy_area": "faq"},
    "escalation_policy.md": {"title": "Escalation Policy", "policy_area": "escalation"},
}


def load_knowledge_base(directory: Path | None = None) -> list[Document]:
    """Load every markdown file in the knowledge base into LangChain Documents.

    Args:
        directory: Folder to read. Defaults to the configured knowledge base.

    Returns:
        One `Document` per file, each carrying source metadata.

    Raises:
        FileNotFoundError: If the directory does not exist or has no .md files.
            We fail loudly, because an empty knowledge base means the agent will
            confidently answer policy questions with nothing to ground it -- the
            exact failure mode this project is designed to prevent.
    """
    kb_dir = directory or settings.knowledge_base_path

    if not kb_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {kb_dir}\n"
            "Expected markdown policy files there. Check KNOWLEDGE_BASE_DIR in .env."
        )

    # sorted() makes ingestion deterministic. Without it, the order depends on
    # the filesystem, which makes debugging "why did chunk 14 change?" painful.
    markdown_files = sorted(kb_dir.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"No .md files found in {kb_dir}. The knowledge base is empty."
        )

    documents: list[Document] = []

    for path in markdown_files:
        # TextLoader is the simplest LangChain loader: read a file, return one
        # Document. We pass encoding explicitly because the default is the OS
        # locale, which silently mangles special characters on some Windows
        # machines.
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()  # -> list[Document], one element for a text file

        catalog_entry = DOCUMENT_CATALOG.get(
            path.name, {"title": path.stem.replace("_", " ").title(), "policy_area": "general"}
        )

        for doc in loaded:
            # TextLoader sets metadata["source"] to the full absolute path.
            # We overwrite it with just the filename: absolute paths differ
            # between your laptop and Docker, which would make citations
            # inconsistent and leak the server's directory layout to customers.
            doc.metadata["source"] = path.name
            doc.metadata["title"] = catalog_entry["title"]
            doc.metadata["policy_area"] = catalog_entry["policy_area"]
            documents.append(doc)

        logger.debug("Loaded %s (%d characters)", path.name, len(loaded[0].page_content))

    total_chars = sum(len(d.page_content) for d in documents)
    logger.info(
        "Loaded %d documents from %s (%d characters total)",
        len(documents),
        kb_dir.name,
        total_chars,
    )
    return documents


if __name__ == "__main__":
    # Run directly to inspect what the loader produces:
    #     python -m app.rag.loader
    from app.config import setup_logging

    setup_logging()
    docs = load_knowledge_base()
    for d in docs:
        print(f"{d.metadata['source']:<28} {len(d.page_content):>6} chars  "
              f"area={d.metadata['policy_area']}")
    print(f"\nFirst 300 characters of {docs[0].metadata['source']}:\n")
    print(docs[0].page_content[:300])
