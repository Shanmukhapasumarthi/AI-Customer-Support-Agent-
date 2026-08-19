"""Tests for the RAG pipeline: loading, splitting, storing, retrieving."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.rag.loader import DOCUMENT_CATALOG, load_knowledge_base
from app.rag.retriever import format_documents, unique_sources
from app.rag.splitter import split_documents


# --- Loading ---------------------------------------------------------------

def test_all_catalogued_documents_load(loaded_documents):
    """Every file we catalogued should actually exist and load."""
    sources = {d.metadata["source"] for d in loaded_documents}
    assert sources == set(DOCUMENT_CATALOG)


def test_documents_carry_citation_metadata(loaded_documents):
    """Without this metadata, citations in Phase 8 are impossible."""
    for doc in loaded_documents:
        assert doc.metadata["source"].endswith(".md")
        assert doc.metadata["title"]
        assert doc.metadata["policy_area"]


def test_source_is_filename_not_absolute_path(loaded_documents):
    """Absolute paths differ between laptop and Docker, and leak server layout."""
    for doc in loaded_documents:
        assert "/" not in doc.metadata["source"]
        assert "\\" not in doc.metadata["source"]


def test_documents_are_not_empty(loaded_documents):
    for doc in loaded_documents:
        assert len(doc.page_content) > 200, f"{doc.metadata['source']} is suspiciously short"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_knowledge_base(tmp_path / "does_not_exist")


def test_empty_directory_raises(tmp_path):
    """An empty knowledge base must fail loudly, not silently produce an
    ungrounded agent."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No .md files"):
        load_knowledge_base(empty)


# --- Splitting -------------------------------------------------------------

def test_splitting_produces_more_chunks_than_documents(loaded_documents, chunks):
    assert len(chunks) > len(loaded_documents)


def test_chunks_respect_size_limit(chunks):
    """RecursiveCharacterTextSplitter can overshoot slightly on an unsplittable
    run of text, so we allow a small margin rather than asserting exact size."""
    from app.config import settings

    for chunk in chunks:
        assert len(chunk.page_content) <= settings.chunk_size + 200


def test_chunks_inherit_parent_metadata(chunks):
    """split_documents() must preserve metadata. split_text() would lose it,
    and with it every citation."""
    for chunk in chunks:
        assert chunk.metadata["source"]
        assert chunk.metadata["title"]
        assert "chunk_index" in chunk.metadata


def test_chunk_indices_are_sequential_per_source(chunks):
    by_source: dict[str, list[int]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata["source"], []).append(
            chunk.metadata["chunk_index"]
        )
    for source, indices in by_source.items():
        assert indices == list(range(len(indices))), f"gaps in {source}"


def test_no_empty_chunks(chunks):
    for chunk in chunks:
        assert chunk.page_content.strip()


def test_every_source_produces_chunks(chunks):
    sources = {c.metadata["source"] for c in chunks}
    assert sources == set(DOCUMENT_CATALOG)


def test_overlap_larger_than_chunk_is_prevented_by_config():
    """The validator in config.py should stop this misconfiguration."""
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, groq_api_key="k", chunk_size=100, chunk_overlap=150)


# --- Vector store plumbing (stub embeddings, no network) -------------------

def test_vector_store_roundtrip(tmp_path, stub_embeddings, chunks):
    """Store chunks and get them back with metadata intact.

    Stub embeddings carry no meaning, so this checks the PLUMBING, not relevance
    ranking. Relevance is covered by the live evaluation harness.
    """
    from langchain_chroma import Chroma

    store = Chroma.from_documents(
        documents=chunks[:20],
        embedding=stub_embeddings,
        collection_name="test_collection",
        persist_directory=str(tmp_path / "vs"),
    )
    assert store._collection.count() == 20

    results = store.similarity_search("refund", k=3)
    assert len(results) == 3
    for doc in results:
        assert doc.metadata["source"].endswith(".md")


def test_retriever_returns_documents(tmp_path, stub_embeddings, chunks):
    from langchain_chroma import Chroma

    store = Chroma.from_documents(
        documents=chunks[:20],
        embedding=stub_embeddings,
        collection_name="test_retriever",
        persist_directory=str(tmp_path / "vs2"),
    )
    # Plain similarity search here: the stub's scores are meaningless, so a
    # score threshold would filter unpredictably.
    retriever = store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("what is the refund policy")
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)


def test_load_missing_vector_store_raises(tmp_path):
    from app.rag.vectorstore import load_vector_store

    with pytest.raises(FileNotFoundError, match="No vector store"):
        load_vector_store(tmp_path / "nothing_here")


# --- Prompt formatting -----------------------------------------------------

def test_format_documents_labels_each_source():
    docs = [
        Document(page_content="Refunds within 30 days.",
                 metadata={"source": "refund_policy.md", "title": "Refund Policy"}),
        Document(page_content="Shipping takes 5-7 days.",
                 metadata={"source": "shipping_policy.md", "title": "Shipping Policy"}),
    ]
    text = format_documents(docs)
    assert "[Source 1: Refund Policy (refund_policy.md)]" in text
    assert "[Source 2: Shipping Policy (shipping_policy.md)]" in text
    assert "Refunds within 30 days." in text


def test_format_documents_signals_emptiness_explicitly():
    """An empty string would look like a formatting bug to the model and invite
    it to fill the gap from memory. A clear statement of absence does not."""
    assert format_documents([]) == "NO RELEVANT DOCUMENTS FOUND IN THE KNOWLEDGE BASE."


def test_unique_sources_deduplicates_preserving_order():
    docs = [
        Document(page_content="a", metadata={"source": "b.md"}),
        Document(page_content="b", metadata={"source": "a.md"}),
        Document(page_content="c", metadata={"source": "b.md"}),
    ]
    assert unique_sources(docs) == ["b.md", "a.md"]
