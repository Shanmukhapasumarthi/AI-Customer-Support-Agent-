from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from app.schemas.response import AnswerSource, Intent, SupportResponse


@pytest.fixture
def client(monkeypatch, seeded_database):
    """A TestClient whose agent is replaced by a deterministic stub."""

    def fake_answer(message: str, session_id: str = "default") -> SupportResponse:
        return SupportResponse(
            answer=f"Stub reply to: {message}",
            intent=Intent.ORDER_STATUS,
            source=AnswerSource.ORDER_DATABASE,
            confidence=0.9,
            requires_human=False,
            tools_used=["get_order_status"],
            session_id=session_id,
            latency_ms=12,
        )

    # Patch where the name is USED, not where it is defined. routes.py did
    # `from app.agents.support_agent import answer_question`, which binds the
    # function into the routes module's namespace. Patching the original module
    # would leave that binding untouched and the stub would never be called.
    # This is the single most common mistake in Python mocking.
    monkeypatch.setattr("app.api.routes.answer_question", fake_answer)

    from app.api.app import create_app

    return TestClient(create_app())


# --- Basic routing ---------------------------------------------------------

def test_root_returns_service_info(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert body["version"]


def test_health_reports_dependencies(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    # A health check that only proves the web server answered is worthless.
    # These fields are the point: they tell you WHICH dependency is missing.
    assert "vector_store_ready" in body
    assert "database_ready" in body
    assert body["database_ready"] is True


def test_openapi_schema_is_generated(client):
    """Free API docs, generated from the same Pydantic models we already wrote."""
    schema = client.get("/openapi.json").json()
    assert "/chat" in schema["paths"]
    assert "SupportResponse" in schema["components"]["schemas"]


# --- /chat -----------------------------------------------------------------

def test_chat_returns_the_full_structured_response(client):
    response = client.post("/chat", json={"message": "Where is ORD1001?",
                                          "session_id": "test-1"})
    assert response.status_code == 200
    body = response.json()
    for field in ["answer", "intent", "source", "confidence", "requires_human",
                  "sources", "tools_used", "session_id"]:
        assert field in body
    assert body["session_id"] == "test-1"


def test_chat_defaults_session_id(client):
    body = client.post("/chat", json={"message": "hello"}).json()
    assert body["session_id"] == "default"


@pytest.mark.parametrize("payload", [
    {},                                   # missing message
    {"message": ""},                      # empty
    {"message": "   "},                   # whitespace only
    {"message": "x" * 3000},              # too long
])
def test_chat_rejects_invalid_payloads(client, payload):
    """422, generated automatically by Pydantic. No hand-written validation."""
    assert client.post("/chat", json=payload).status_code == 422


# --- Raw data endpoints ----------------------------------------------------

def test_get_order_returns_data(client):
    body = client.get("/orders/ORD1001").json()
    assert body["order_id"] == "ORD1001"
    assert body["status"] == "Shipped"


def test_get_order_is_case_insensitive(client):
    assert client.get("/orders/ord1001").status_code == 200


def test_get_missing_order_returns_404(client):
    """An API for machines uses status codes. The AGENT uses sentences instead,
    because it is talking to a human -- the two are deliberately different."""
    response = client.get("/orders/ORD9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_product_returns_current_price(client):
    body = client.get("/products/PROD001").json()
    assert body["price"] == "199.99"


def test_get_missing_product_returns_404(client):
    assert client.get("/products/PROD999").status_code == 404


# --- Session management ----------------------------------------------------

def test_session_info_starts_empty(client):
    body = client.get("/sessions/brand-new-session").json()
    assert body["message_count"] == 0


def test_delete_session_is_idempotent(client):
    """Deleting a session that never existed must not error -- the UI's 'New
    conversation' button should always work."""
    assert client.delete("/sessions/never-existed").status_code == 200
