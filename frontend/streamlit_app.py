from __future__ import annotations
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 90  # generous: an agent run can involve several LLM calls

st.set_page_config(page_title="NimbusCart Support", page_icon="*", layout="centered")

# --- Friendly labels for the machine-readable enum values -------------------
INTENT_LABELS = {
    "order_status": "Order Status",
    "order_cancellation": "Order Cancellation",
    "refund_policy": "Refund Policy",
    "refund_request": "Refund Request",
    "return_policy": "Return Policy",
    "shipping_policy": "Shipping Policy",
    "warranty_policy": "Warranty Policy",
    "payment_policy": "Payment Policy",
    "product_info": "Product Info",
    "product_comparison": "Product Comparison",
    "complaint": "Complaint",
    "escalation": "Escalation",
    "greeting": "Greeting",
    "unknown": "Unknown",
}

SOURCE_LABELS = {
    "knowledge_base": "Knowledge Base",
    "order_database": "Order Database",
    "both": "Knowledge Base + Order Database",
    "none": "No source",
}


def init_state() -> None:
    """Create session_state keys on first run only."""
    if "session_id" not in st.session_state:
        # A random ID per browser session keeps different users' agent memory
        # separate. Reuse the same ID and you would share one conversation.
        st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []


def call_api(message: str) -> dict:
    """POST the message to /chat and return the structured response."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "session_id": st.session_state.session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": f"I can't reach the support service at {API_URL}. "
                      "Is the API running? Start it with: python main.py",
            "intent": "unknown", "source": "none", "confidence": 0.0,
            "requires_human": True,
            "escalation_reason": "The backend API is unreachable.",
            "sources": [], "tools_used": [],
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "That took too long to process. Please try again.",
            "intent": "unknown", "source": "none", "confidence": 0.0,
            "requires_human": True,
            "escalation_reason": "The request timed out.",
            "sources": [], "tools_used": [],
        }
    except requests.exceptions.HTTPError as exc:
        return {
            "answer": f"The support service returned an error: {exc}",
            "intent": "unknown", "source": "none", "confidence": 0.0,
            "requires_human": True, "escalation_reason": "Backend error.",
            "sources": [], "tools_used": [],
        }


def render_metadata(data: dict) -> None:
    """Show the structured fields under an answer.

    THIS IS THE POINT OF STRUCTURED OUTPUT. Everything below comes from the
    Pydantic response model, not from parsing the answer text. That is what
    makes a source badge, an intent label, and an escalation banner possible.
    """
    if data.get("requires_human"):
        st.warning(
            f"**Escalated to a human specialist.** "
            f"{data.get('escalation_reason', '')}",
            icon=None,
        )

    source_label = SOURCE_LABELS.get(data.get("source", "none"), "Unknown")
    intent_label = INTENT_LABELS.get(data.get("intent", "unknown"), "Unknown")
    confidence = data.get("confidence", 0.0)

    col1, col2, col3 = st.columns(3)
    col1.caption(f"**Source:** {source_label}")
    col2.caption(f"**Intent:** {intent_label}")
    col3.caption(f"**Confidence:** {confidence:.0%}")

    sources = data.get("sources") or []
    tools = data.get("tools_used") or []
    if sources or tools:
        with st.expander("How this answer was produced"):
            if tools:
                st.write("**Tools called:**")
                for tool in tools:
                    st.write(f"- `{tool}`")
            if sources:
                st.write("**Documents retrieved:**")
                for source in sources:
                    st.write(f"- `{source}`")
            latency = data.get("latency_ms")
            if latency:
                st.write(f"**Response time:** {latency} ms")


def sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.code(st.session_state.session_id, language=None)
        st.caption(
            "The agent remembers this conversation. Say \"my order is ORD1001\", "
            "then ask \"when will it arrive?\" to see memory working."
        )

        if st.button("New conversation", use_container_width=True):
            # Clear BOTH sides: the agent's memory on the server AND the UI's
            # displayed history. Clearing only one leaves them out of sync,
            # which looks like a bug to the user.
            try:
                requests.delete(
                    f"{API_URL}/sessions/{st.session_state.session_id}", timeout=10
                )
            except requests.RequestException:
                pass
            st.session_state.messages = []
            st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"
            st.rerun()

        st.divider()
        st.header("Service status")
        try:
            health = requests.get(f"{API_URL}/health", timeout=10).json()
            ok = health.get("status") == "ok"
            st.write(f"**Status:** {'Healthy' if ok else 'Degraded'}")
            st.write(f"**Model:** `{health.get('llm_model', '?')}`")
            st.write(f"**Chunks indexed:** {health.get('chunks_indexed', '?')}")
            rows = health.get("database_rows") or {}
            st.write(f"**Orders in DB:** {rows.get('orders', '?')}")
            if health.get("detail"):
                st.caption(health["detail"])
        except requests.RequestException:
            st.error(f"API unreachable at {API_URL}")

        st.divider()
        st.header("Try asking")
        for example in [
            "What is your refund policy?",
            "Where is order ORD1001?",
            "Can I return an item after 15 days?",
            "What is the price of PROD001?",
            "What's the refund amount for ORD1002?",
            "I want to cancel ORD1004.",
            "How is the X1 different from the Mini?",
            "My payment was charged twice.",
        ]:
            st.caption(f"- {example}")


def main() -> None:
    init_state()

    st.title("NimbusCart Customer Support")
    st.caption("AI assistant with access to our policy documents and order system.")

    sidebar()

    # Redraw the whole conversation. Streamlit rebuilds the page from scratch on
    # every interaction, so history must be replayed from session_state.
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("data"):
                render_metadata(entry["data"])

    prompt = st.chat_input("Ask about an order, a policy, or a product...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking our records..."):
            data = call_api(prompt)
        answer = data.get("answer", "Sorry, something went wrong.")
        st.markdown(answer)
        render_metadata(data)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "data": data}
    )


if __name__ == "__main__":
    main()
