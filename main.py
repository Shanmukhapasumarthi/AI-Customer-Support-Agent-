from __future__ import annotations
import argparse
import sys

from app.config import PROJECT_ROOT, settings, setup_logging

logger = setup_logging()


def check_config() -> None:
    logger.info("=" * 64)
    logger.info("CONFIGURATION")
    logger.info("=" * 64)
    logger.info("App name        : %s", settings.app_name)
    logger.info("Project root    : %s", PROJECT_ROOT)
    logger.info("LLM model       : %s", settings.groq_model)
    logger.info("Temperature     : %s", settings.llm_temperature)
    key = settings.groq_api_key
    # Never log a full secret: enough to confirm the right key loaded, not
    # enough to leak through a screenshot or a shipped log file.
    logger.info("Groq API key    : %s...%s (%d chars)", key[:6], key[-4:], len(key))
    logger.info("Embedding model : %s", settings.embedding_model)
    logger.info("Chunk size      : %s (overlap %s)", settings.chunk_size,
                settings.chunk_overlap)
    logger.info("Retriever top-k : %s", settings.retriever_top_k)


def check_llm() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.llm import get_llm

    logger.info("=" * 64)
    logger.info("LLM CONNECTIVITY")
    logger.info("=" * 64)

    # A chat model consumes a LIST OF MESSAGES, not a raw string.
    # SystemMessage = instructions (highest authority). HumanMessage = the user.
    response = get_llm().invoke([
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="Reply with exactly: SETUP OK"),
    ])
    logger.info("Model replied   : %s", response.content.strip())


def check_tool_calling() -> None:
    """The single most important capability check in this project.

    The agent only works if the model can emit structured tool calls. If this
    fails, no amount of prompting helps -- you must change the model.
    """
    from pydantic import BaseModel, Field

    from app.llm import get_llm

    logger.info("=" * 64)
    logger.info("TOOL-CALLING CAPABILITY")
    logger.info("=" * 64)

    class OrderLookup(BaseModel):
        """Look up the status of a customer order."""
        order_id: str = Field(description="The order id, e.g. ORD1001")

    # bind_tools() sends the model a JSON schema generated from this Pydantic
    # class -- its field names, types, and docstring.
    response = get_llm().bind_tools([OrderLookup]).invoke("Where is my order ORD1001?")

    if not response.tool_calls:
        raise RuntimeError(
            f"Model {settings.groq_model!r} did not emit a tool call. "
            "Set a tool-calling-capable GROQ_MODEL in .env."
        )
    call = response.tool_calls[0]
    logger.info("Tool selected   : %s  args=%s", call["name"], call["args"])
    logger.info("Tool calling is SUPPORTED.")


def check_data_stores() -> None:
    logger.info("=" * 64)
    logger.info("DATA STORES")
    logger.info("=" * 64)

    from app.rag.vectorstore import load_vector_store

    store = load_vector_store()
    logger.info("Vector store    : %d chunks indexed", store._collection.count())

    from app.database.database import database_stats

    logger.info("Database        : %s", database_stats())


def run_checks() -> int:
    try:
        check_config()
        check_llm()
        check_tool_calling()
        check_data_stores()
    except FileNotFoundError as exc:
        logger.error("Setup incomplete: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Check FAILED: %s: %s", type(exc).__name__, exc)
        return 1

    logger.info("=" * 64)
    logger.info("ALL CHECKS PASSED. Start the server with: python main.py")
    logger.info("=" * 64)
    return 0


def serve() -> None:
    import uvicorn

    logger.info("Starting API on http://%s:%d (docs at /docs)",
                settings.api_host, settings.api_port)
    uvicorn.run(
        "app.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=settings.app_name)
    parser.add_argument("--check", action="store_true",
                        help="Verify setup and exit without starting the server.")
    args = parser.parse_args()

    if args.check:
        sys.exit(run_checks())
    serve()
