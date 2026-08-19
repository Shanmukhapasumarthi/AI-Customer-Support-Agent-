from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, status

from app.agents.memory import clear_session, session_message_count
from app.agents.support_agent import answer_question, reset_agent
from app.config import settings
from app.database.database import database_stats, fetch_order, fetch_product
from app.schemas.response import ChatRequest, HealthResponse, SupportResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report whether the service and its two data stores are usable.

    A health check that only returns {"status": "ok"} is close to worthless -- it
    proves the web server is up, which you already knew because it answered. A
    useful one checks the DEPENDENCIES: is the vector store built? Is the
    database seeded? Those are the two things that are actually likely to be
    missing, and this catches "I forgot to run the ingest script" immediately.
    """
    vector_ready = False
    chunks: int | None = None
    detail: str | None = None

    try:
        from app.rag.vectorstore import load_vector_store

        store = load_vector_store()
        chunks = store._collection.count()
        vector_ready = chunks > 0
    except Exception as exc:  # noqa: BLE001
        detail = f"Vector store unavailable: {exc}. Run: python -m scripts.ingest"

    db_ready = False
    rows: dict[str, int] | None = None
    try:
        rows = database_stats()
        db_ready = rows["orders"] > 0
        if not db_ready:
            detail = ((detail + " | ") if detail else "") + (
                "Database is empty. Run: python -m scripts.seed_database"
            )
    except Exception as exc:  # noqa: BLE001
        detail = ((detail + " | ") if detail else "") + f"Database unavailable: {exc}"

    return HealthResponse(
        status="ok" if (vector_ready and db_ready) else "degraded",
        app_name=settings.app_name,
        llm_model=settings.groq_model,
        vector_store_ready=vector_ready,
        database_ready=db_ready,
        chunks_indexed=chunks,
        database_rows=rows,
        detail=detail,
    )


@router.post("/chat", response_model=SupportResponse, tags=["chat"])
def chat(request: ChatRequest) -> SupportResponse:
    """Main endpoint. Send a customer message, get a structured answer.

    Note there is no try/except returning a 500 here. `answer_question` already
    converts any internal failure into an escalated SupportResponse. A support
    product should degrade to "a human will help you" rather than to a 500 page.
    """
    logger.info("POST /chat session=%s message=%r",
                request.session_id, request.message[:80])
    return answer_question(message=request.message, session_id=request.session_id)


@router.get("/orders/{order_id}", tags=["data"])
def get_order(order_id: str) -> dict:
    """Raw order lookup, bypassing the agent.

    Useful for debugging ("is the agent wrong, or is the data wrong?") and for
    any client that wants the data without the LLM. Note this returns a real 404
    rather than the agent's soft "not found" -- an API for machines should use
    status codes, while an agent talking to a human should use sentences.
    """
    order = fetch_order(order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id.upper()} not found",
        )
    return order


@router.get("/products/{product_id}", tags=["data"])
def get_product(product_id: str) -> dict:
    """Raw product lookup, bypassing the agent."""
    product = fetch_product(product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id.upper()} not found",
        )
    return product


@router.post("/knowledge/reload", tags=["system"])
def reload_knowledge() -> dict:
    """Re-ingest the knowledge base without restarting the server.

    WHY THIS ENDPOINT EXISTS: policies change. Without it, updating
    refund_policy.md means a full redeploy. With it, edit the file and POST here.

    The three steps matter and must happen in this order:
      1. Rebuild the vector store from the markdown files.
      2. Clear the cached retriever, which still points at the OLD collection.
      3. Clear the cached agent, which holds a tool bound to the old retriever.
    Skip step 2 or 3 and the endpoint appears to succeed while the agent keeps
    serving stale policy -- a genuinely confusing bug to chase.
    """
    try:
        from app.tools.knowledge_tools import reset_retriever
        from scripts.ingest import ingest

        chunk_count = ingest(verify=False)
        reset_retriever()
        reset_agent()

        logger.info("Knowledge base reloaded: %d chunks", chunk_count)
        return {
            "status": "reloaded",
            "chunks_indexed": chunk_count,
            "message": "Vector store rebuilt and caches cleared.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Knowledge reload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reload failed: {exc}",
        ) from exc


@router.delete("/sessions/{session_id}", tags=["chat"])
def delete_session(session_id: str) -> dict:
    """Clear one conversation's memory. Backs the 'New conversation' button."""
    existed = clear_session(session_id)
    return {"session_id": session_id, "cleared": existed}


@router.get("/sessions/{session_id}", tags=["chat"])
def session_info(session_id: str) -> dict:
    """How many messages are stored for a session."""
    return {
        "session_id": session_id,
        "message_count": session_message_count(session_id),
    }
