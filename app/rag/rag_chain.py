from __future__ import annotations
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from app.llm import get_llm
from app.rag.retriever import build_retriever, format_documents, unique_sources

logger = logging.getLogger(__name__)


RAG_SYSTEM_PROMPT = """You are a customer support assistant for NimbusCart, an \
online retailer.

Answer the customer's question using ONLY the context below. The context contains \
excerpts from NimbusCart's official policy documents.

STRICT RULES:
1. Use only facts present in the context. Do not add information from general \
knowledge about how online stores usually work.
2. If the context does not contain the answer, say exactly: "I don't have that \
information in my knowledge base." Do not guess.
3. Never invent numbers, timeframes, prices, or policy details.
4. Quote specific numbers from the context when they answer the question \
(for example "30 calendar days", "5 to 7 business days").
5. Be concise: two to four sentences unless the customer asks for detail.
6. Write in plain, warm language. No markdown headings, no bullet lists unless \
comparing several items.

CONTEXT:
{context}"""


def build_rag_chain(retriever: Runnable | None = None) -> Runnable:
    """Compose the retrieval-augmented generation chain.

    Returns:
        A Runnable taking a question string and returning an answer string.
    """
    retriever = retriever or build_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    # ---------------------------------------------------------------------
    # THE CHAIN, read top to bottom.
    #
    # RunnableParallel runs both branches on the SAME input (the question) and
    # produces a dict. `retriever | format_documents` fetches chunks and turns
    # them into the prompt text block. RunnablePassthrough-style lambda carries
    # the raw question through untouched.
    #
    # The resulting dict {"context": ..., "question": ...} is exactly what the
    # prompt template's placeholders need. That is why the keys are named this
    # way -- they are matched to `{context}` and `{question}` by name.
    # ---------------------------------------------------------------------
    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_documents),
            question=RunnableLambda(lambda q: q),
        )
        | prompt          # dict  -> ChatPromptValue (a list of messages)
        | get_llm()       # messages -> AIMessage
        | StrOutputParser()  # AIMessage -> plain str (grabs .content)
    )

    return chain


def answer_with_sources(
    question: str, retriever: Runnable | None = None
) -> dict[str, Any]:
    """Run RAG and return the answer TOGETHER WITH the evidence.

    `build_rag_chain()` returns only a string, which throws away the documents.
    For a support product that is not good enough: we need to show the customer
    which policy backed the answer, and we need the similarity scores to compute
    a confidence value in Phase 14. So this function retrieves once, explicitly,
    and keeps everything.

    Returns:
        dict with keys: answer, sources, num_chunks, grounded.
    """
    retriever = retriever or build_retriever()

    documents: list[Document] = retriever.invoke(question)

    if not documents:
        # The retriever's score threshold rejected everything. Short-circuit
        # WITHOUT calling the LLM: there is nothing to ground an answer in, and
        # calling the model anyway is how hallucinations happen.
        logger.info("No chunks passed the threshold for %r", question)
        return {
            "answer": "I don't have that information in my knowledge base.",
            "sources": [],
            "num_chunks": 0,
            "grounded": False,
        }

    prompt = ChatPromptTemplate.from_messages(
        [("system", RAG_SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | get_llm() | StrOutputParser()

    answer = chain.invoke(
        {"context": format_documents(documents), "question": question}
    )

    return {
        "answer": answer,
        "sources": unique_sources(documents),
        "num_chunks": len(documents),
        "grounded": True,
    }


if __name__ == "__main__":
    #     python -m app.rag.rag_chain
    from app.config import setup_logging

    setup_logging()

    for question in [
        "What is your refund policy?",
        "How long does shipping take?",
        "Can I return an item after 15 days?",
        "Who is the CEO of NimbusCart?",
    ]:
        result = answer_with_sources(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"   sources={result['sources']} chunks={result['num_chunks']}")
