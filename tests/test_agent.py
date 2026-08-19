from __future__ import annotations
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.agents.memory import (
    MAX_HISTORY_MESSAGES,
    WindowedChatMessageHistory,
    clear_session,
    get_session_history,
    session_message_count,
)
from app.agents.support_agent import (
    _apply_confidence_floor,
    _derive_source,
    _extract_tools_used,
    _fallback_classification,
)
from app.schemas.response import (
    AnswerSource,
    ChatRequest,
    Intent,
    SupportResponse,
)


# --- Memory ----------------------------------------------------------------

def test_history_stores_messages():
    history = WindowedChatMessageHistory("t1")
    history.add_messages([HumanMessage(content="hi"), AIMessage(content="hello")])
    assert len(history.messages) == 2


def test_history_window_drops_oldest_first():
    """Trimming must keep RECENT context. Dropping the newest would be the exact
    opposite of what a conversation needs."""
    history = WindowedChatMessageHistory("t2", max_messages=4)
    for i in range(10):
        history.add_messages([HumanMessage(content=f"msg{i}")])
    assert len(history.messages) == 4
    assert history.messages[-1].content == "msg9"
    assert history.messages[0].content == "msg6"


def test_history_default_window():
    history = WindowedChatMessageHistory("t3")
    assert history.max_messages == MAX_HISTORY_MESSAGES


def test_messages_property_returns_a_copy():
    """Callers must not be able to mutate our internal list."""
    history = WindowedChatMessageHistory("t4")
    history.add_messages([HumanMessage(content="a")])
    history.messages.append(HumanMessage(content="injected"))
    assert len(history.messages) == 1


def test_sessions_are_isolated():
    """The property that makes one agent object safe for many users."""
    a = get_session_history("user-a")
    b = get_session_history("user-b")
    a.add_messages([HumanMessage(content="secret for A")])

    assert session_message_count("user-a") == 1
    assert session_message_count("user-b") == 0
    assert len(b.messages) == 0

    clear_session("user-a")
    clear_session("user-b")


def test_same_session_id_returns_same_history():
    first = get_session_history("stable")
    second = get_session_history("stable")
    assert first is second
    clear_session("stable")


def test_clear_session_reports_whether_it_existed():
    get_session_history("temp")
    assert clear_session("temp") is True
    assert clear_session("temp") is False


# --- Provenance helpers ----------------------------------------------------

class _FakeAction:
    def __init__(self, tool: str):
        self.tool = tool


def test_extract_tools_used_reads_real_trace():
    steps = [(_FakeAction("search_knowledge_base"), "..."),
             (_FakeAction("get_order_status"), "...")]
    assert _extract_tools_used(steps) == ["search_knowledge_base", "get_order_status"]


def test_extract_tools_used_deduplicates():
    steps = [(_FakeAction("get_order_status"), "a"), (_FakeAction("get_order_status"), "b")]
    assert _extract_tools_used(steps) == ["get_order_status"]


def test_extract_tools_used_handles_empty_trace():
    assert _extract_tools_used([]) == []


@pytest.mark.parametrize("tools,expected", [
    (["search_knowledge_base"], AnswerSource.KNOWLEDGE_BASE),
    (["get_order_status"], AnswerSource.ORDER_DATABASE),
    (["calculate_refund_amount"], AnswerSource.ORDER_DATABASE),
    (["search_knowledge_base", "get_order_status"], AnswerSource.BOTH),
    ([], AnswerSource.NONE),
])
def test_source_derived_from_observed_tools(tools, expected):
    """`source` comes from what ACTUALLY ran, never from the model's claim."""
    assert _derive_source(tools) == expected


# --- Confidence guards -----------------------------------------------------

def test_confidence_floored_when_answer_is_a_non_answer():
    """A classifier LLM sometimes rates a polite 'I don't know' as confident --
    it IS a well-formed reply. A deterministic string check cannot be fooled."""
    answer = "I don't have that information in my knowledge base."
    assert _apply_confidence_floor(answer, 0.95) <= 0.3


def test_confidence_untouched_for_a_real_answer():
    answer = "Your order ORD1001 has shipped and arrives on 21 August."
    assert _apply_confidence_floor(answer, 0.95) == 0.95


def test_fallback_classification_escalates_on_uncertainty():
    result = _fallback_classification("I couldn't find that order.")
    assert result.requires_human is True
    assert result.confidence < 0.3


def test_fallback_classification_does_not_escalate_normal_answers():
    result = _fallback_classification("Refunds take 5 to 7 business days.")
    assert result.requires_human is False


# --- Schemas ---------------------------------------------------------------

def test_support_response_minimal_construction():
    r = SupportResponse(answer="hello")
    assert r.intent == Intent.UNKNOWN
    assert r.source == AnswerSource.NONE
    assert r.requires_human is False


def test_confidence_must_be_within_range():
    with pytest.raises(ValidationError):
        SupportResponse(answer="x", confidence=1.5)
    with pytest.raises(ValidationError):
        SupportResponse(answer="x", confidence=-0.1)


def test_escalation_always_carries_a_reason():
    """A human picking this out of a queue needs to know why it arrived."""
    r = SupportResponse(answer="x", requires_human=True)
    assert r.escalation_reason


def test_invalid_intent_rejected():
    """A closed enum is what makes analytics possible."""
    with pytest.raises(ValidationError):
        SupportResponse(answer="x", intent="banana")


def test_enums_serialise_to_plain_strings():
    payload = SupportResponse(
        answer="x", intent=Intent.ORDER_STATUS, source=AnswerSource.ORDER_DATABASE
    ).model_dump(mode="json")
    assert payload["intent"] == "order_status"
    assert payload["source"] == "order_database"


def test_chat_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_trims_whitespace():
    assert ChatRequest(message="  hello  ").message == "hello"


def test_chat_request_defaults_session_id():
    assert ChatRequest(message="hi").session_id == "default"


def test_chat_request_rejects_oversized_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 3000)


# --- Prompt safety ---------------------------------------------------------

def test_system_prompt_contains_the_core_grounding_rules():
    from app.agents.prompts import SUPPORT_AGENT_SYSTEM_PROMPT as prompt

    lowered = prompt.lower()
    assert "i don't have that information in my knowledge base" in lowered
    assert "never invent" in lowered
    assert "never claim you used a tool you did not use" in lowered
    assert "cannot" in lowered  # the "you cannot issue a refund" section


def test_agent_prompt_has_all_four_required_slots():
    """agent_scratchpad is the one people forget. Without it the agent never
    sees its own tool results and loops until max_iterations."""
    from app.agents.prompts import build_agent_prompt

    variables = set(build_agent_prompt().input_variables) | set(
        build_agent_prompt().optional_variables
    )
    assert {"input", "chat_history", "agent_scratchpad"} <= variables
