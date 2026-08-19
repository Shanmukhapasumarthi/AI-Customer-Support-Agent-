from __future__ import annotations
import logging
import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.agents.memory import get_session_history
from app.agents.prompts import SUPPORT_AGENT_SYSTEM_PROMPT, build_classifier_prompt
from app.llm import get_llm
from app.schemas.response import (
    AnswerSource,
    Intent,
    ResponseClassification,
    SupportResponse,
)
from app.tools import ALL_TOOLS
from app.tools.knowledge_tools import clear_last_sources, get_last_sources

logger = logging.getLogger(__name__)

# Tools that read the knowledge base vs. tools that read the database. Used to
# derive the `source` field from OBSERVED behaviour rather than the model's claim.
KNOWLEDGE_TOOL_NAMES = {"search_knowledge_base"}
DATABASE_TOOL_NAMES = {
    "get_order_status",
    "get_customer_orders",
    "get_product_information",
    "search_products",
    "calculate_refund_amount",
    "check_cancellation_eligibility",
}

# Phrases that indicate the agent declined to answer. Used to floor the
# confidence score, because a classifier LLM sometimes rates a polite
# "I don't know" as a confident answer -- it IS a well-formed reply, after all.
UNCERTAINTY_MARKERS = (
    "i don't have that information",
    "i do not have that information",
    "i don't have enough information",
    "not in my knowledge base",
    "couldn't find",
    "could not find",
    "no order with",
    "unable to find",
)

_agent: Any | None = None
_conversational_agent: Runnable | None = None


# ---------------------------------------------------------------------------
# PHASE 11 + 12: building the agent
# ---------------------------------------------------------------------------

def build_agent_executor() -> Any:
    """Construct the LangChain 1.x tool-calling agent.

    `create_agent()` returns the complete agent runtime. Unlike the old
    AgentExecutor API, there is no separate executor object.
    """
    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SUPPORT_AGENT_SYSTEM_PROMPT,
    )

    logger.info(
        "Agent built with %d tools: %s",
        len(ALL_TOOLS),
        [t.name for t in ALL_TOOLS],
    )
    return agent


def _content_to_text(content: Any) -> str:
    """Convert an AI message's content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if value:
                    parts.append(str(value))
        return "".join(parts)

    return str(content or "")


def _extract_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    """Adapt LangChain 1.x agent state to the application's old result shape.

    LangChain 1.x returns a state containing a `messages` list rather than the
    old AgentExecutor keys (`output`, `intermediate_steps`).

    Tool calls are reconstructed from AIMessage.tool_calls so provenance still
    comes from observed agent behaviour rather than the model's prose.
    """
    messages = result.get("messages", [])
    tools_used: list[str] = []
    intermediate_steps: list[Any] = []
    answer = ""

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                name = tool_call.get("name")
                if name and name not in tools_used:
                    tools_used.append(name)

                intermediate_steps.append(
                    {
                        "tool": name,
                        "tool_call_id": tool_call.get("id"),
                        "args": tool_call.get("args", {}),
                    }
                )

            content = _content_to_text(message.content)
            if content:
                answer = content

    # The last ToolMessage is not the final answer; the last AIMessage with
    # textual content is the assistant's response.
    return {
        "output": answer,
        "intermediate_steps": intermediate_steps,
        "tools_used": tools_used,
    }


def _invoke_agent_for_history(inputs: dict[str, Any]) -> dict[str, Any]:
    """Invoke the LangChain 1.x agent with the session history."""
    agent = _get_raw_agent()
    message = inputs.get("input", "")
    history = inputs.get("chat_history", [])

    messages = list(history)
    messages.append({"role": "user", "content": message})

    result = agent.invoke({"messages": messages})
    return _extract_agent_result(result)


def _get_raw_agent() -> Any:
    """Return the shared LangChain 1.x agent, building it once per process."""
    global _agent
    if _agent is None:
        _agent = build_agent_executor()
    return _agent



# ---------------------------------------------------------------------------
# PHASE 13: wrapping the agent with conversation memory
# ---------------------------------------------------------------------------

def build_conversational_agent() -> Runnable:
    """Wrap the LangChain 1.x agent with session-based conversation memory.

    The adapter exposes the same input/output keys used by the existing API:
        input -> output
    while the underlying LangChain 1.x agent uses a `messages` state.
    """
    runnable = RunnableLambda(_invoke_agent_for_history)

    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )



def get_agent() -> Runnable:
    """Return the shared conversational agent, building it once per process."""
    global _conversational_agent
    if _conversational_agent is None:
        _conversational_agent = build_conversational_agent()
    return _conversational_agent


def reset_agent() -> None:
    """Force a rebuild on next use. Called after the knowledge base reloads, so
    the agent picks up the new retriever."""
    global _agent, _conversational_agent
    _agent = None
    _conversational_agent = None


# ---------------------------------------------------------------------------
# PHASE 14: turning the agent's prose into a validated structured response
# ---------------------------------------------------------------------------

def _extract_tools_used(intermediate_steps: list[Any]) -> list[str]:
    """Read actual tool names from the LangChain 1.x trace adapter."""
    names: list[str] = []

    for step in intermediate_steps:
        if isinstance(step, dict):
            name = step.get("tool")
        else:
            action = (
                step[0]
                if isinstance(step, (tuple, list)) and step
                else None
            )
            name = getattr(action, "tool", None)

        if name and name not in names:
            names.append(name)

    return names



def _derive_source(tools_used: list[str]) -> AnswerSource:
    """Map observed tool usage onto the `source` field."""
    used_kb = any(t in KNOWLEDGE_TOOL_NAMES for t in tools_used)
    used_db = any(t in DATABASE_TOOL_NAMES for t in tools_used)

    if used_kb and used_db:
        return AnswerSource.BOTH
    if used_kb:
        return AnswerSource.KNOWLEDGE_BASE
    if used_db:
        return AnswerSource.ORDER_DATABASE
    return AnswerSource.NONE


def _classify(message: str, answer: str, tools_used: list[str]) -> ResponseClassification:
    """Second LLM call: label the exchange with intent, confidence, escalation.

    `with_structured_output(Model)` is the important line. Under the hood it
    binds the Pydantic model as a tool schema and forces the model to call it, so
    the reply is guaranteed to be parseable JSON matching our class. No regex, no
    "please respond in JSON" pleading, no try/except around json.loads.
    """
    classifier_llm = get_llm(temperature=0.0).with_structured_output(
        ResponseClassification
    )
    chain = build_classifier_prompt() | classifier_llm

    result = chain.invoke({
        "message": message,
        "answer": answer,
        "tools_used": ", ".join(tools_used) or "none",
    })
    return result  # type: ignore[return-value]


def _fallback_classification(answer: str) -> ResponseClassification:
    """Used when the classifier call itself fails.

    A support agent must degrade gracefully: if labelling breaks, the customer
    should still get their answer. We return conservative metadata and, when the
    answer looks like a non-answer, route to a human rather than dropping it.
    """
    uncertain = any(marker in answer.lower() for marker in UNCERTAINTY_MARKERS)
    return ResponseClassification(
        intent=Intent.UNKNOWN,
        confidence=0.2 if uncertain else 0.6,
        requires_human=uncertain,
        escalation_reason=(
            "The assistant could not find this information and response "
            "classification was unavailable."
        ) if uncertain else None,
    )


def _apply_confidence_floor(answer: str, confidence: float) -> float:
    """Cap confidence when the answer is visibly a non-answer.

    Defence in depth again: the classifier is another LLM and can be wrong. A
    deterministic string check cannot be talked out of its opinion.
    """
    if any(marker in answer.lower() for marker in UNCERTAINTY_MARKERS):
        return min(confidence, 0.3)
    return confidence


# ---------------------------------------------------------------------------
# THE PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def answer_question(message: str, session_id: str = "default") -> SupportResponse:
    """Run one customer message through the full pipeline.

    Steps:
      1. Clear per-request source tracking.
      2. Run the agent (it chooses tools and may call several).
      3. Read which tools ACTUALLY ran and which documents were read.
      4. Classify the exchange into intent / confidence / escalation.
      5. Apply deterministic guards on top of the classifier.
      6. Return a validated SupportResponse.

    Never raises for ordinary failures: a customer-facing service must always
    return something, so errors become an escalated response instead.
    """
    started = time.perf_counter()

    # Source tracking is module-level state in knowledge_tools. Clearing it here
    # guarantees this request's citations cannot include a previous request's
    # documents.
    clear_last_sources()

    try:
        result = get_agent().invoke(
            {"input": message},
            # `configurable.session_id` is how RunnableWithMessageHistory knows
            # WHICH conversation this belongs to. Change the id, get a different
            # memory. This is what makes one agent object safe for many users.
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run failed for session %s", session_id)
        return SupportResponse(
            answer=(
                "I'm sorry, I ran into a technical problem and couldn't process "
                "that. I'm passing you to a human specialist who can help."
            ),
            intent=Intent.ESCALATION,
            source=AnswerSource.NONE,
            confidence=0.0,
            requires_human=True,
            escalation_reason=f"Technical failure: {type(exc).__name__}",
            session_id=session_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    answer = (result.get("output") or "").strip()
    if not answer:
        answer = "I'm sorry, I wasn't able to produce an answer to that."

    tools_used = _extract_tools_used(result.get("intermediate_steps", []))
    kb_sources = get_last_sources()

    try:
        classification = _classify(message, answer, tools_used)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classifier failed (%s); using fallback.", type(exc).__name__)
        classification = _fallback_classification(answer)

    confidence = _apply_confidence_floor(answer, classification.confidence)

    # Deterministic escalation guard. If confidence ended up very low, escalate
    # regardless of what the classifier decided. A customer left with an
    # unanswered question and no route to a human is the worst outcome here.
    requires_human = classification.requires_human or confidence < 0.25
    escalation_reason = classification.escalation_reason
    if requires_human and not escalation_reason:
        escalation_reason = (
            "The assistant could not answer this confidently from the knowledge "
            "base or the order database."
        )

    response = SupportResponse(
        answer=answer,
        intent=classification.intent,
        source=_derive_source(tools_used),  # observed, not claimed
        confidence=round(confidence, 2),
        requires_human=requires_human,
        escalation_reason=escalation_reason if requires_human else None,
        sources=kb_sources,
        tools_used=tools_used,
        session_id=session_id,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )

    logger.info(
        "session=%s intent=%s source=%s conf=%.2f human=%s tools=%s %dms",
        session_id, response.intent.value, response.source.value,
        response.confidence, response.requires_human, tools_used,
        response.latency_ms,
    )
    return response


if __name__ == "__main__":
    #     python -m app.agents.support_agent
    from app.config import setup_logging

    setup_logging()

    print("\n" + "=" * 70)
    print("SINGLE-TURN QUESTIONS")
    print("=" * 70)
    for q in [
        "What is your refund policy?",
        "Where is order ORD1001?",
        "What is the price of PROD001?",
        "Who is the CEO of NimbusCart?",
        "My payment was charged twice and I want this fixed now.",
    ]:
        r = answer_question(q, session_id="demo")
        print(f"\nQ: {q}")
        print(f"A: {r.answer}")
        print(f"   intent={r.intent.value} source={r.source.value} "
              f"conf={r.confidence} human={r.requires_human} tools={r.tools_used}")

    print("\n" + "=" * 70)
    print("MULTI-TURN CONVERSATION (memory test)")
    print("=" * 70)
    for q in ["My order is ORD1001.", "When will it arrive?", "What product was it?"]:
        r = answer_question(q, session_id="memory-demo")
        print(f"\nQ: {q}")
        print(f"A: {r.answer}")