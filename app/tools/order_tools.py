"""
PHASE 10 - LANGCHAIN TOOLS (ORDERS)

WHAT IS A TOOL?
A Python function the LLM is allowed to call. LangChain wraps it so that the
model sees a JSON schema describing the function -- its name, its arguments, and
crucially its DESCRIPTION -- and can emit a request to run it.

HOW TOOL CALLING ACTUALLY WORKS (the mechanics, step by step)
1. We call `llm.bind_tools([...])`. LangChain converts each tool into a JSON
   schema and sends it alongside the user's message.
2. The model reads the customer's question and every tool description, and
   decides whether one of them fits. If so it does NOT return prose. It returns
   a structured tool call:
       {"name": "get_order_status", "args": {"order_id": "ORD1001"}}
3. Our code -- specifically the AgentExecutor in Phase 11 -- actually runs the
   Python function. The LLM cannot execute anything itself; it can only ask.
4. The return value is appended to the conversation as a `ToolMessage`.
5. The model is called AGAIN, now with the tool result in context, and writes
   the natural-language answer.

The loop is: think -> call tool -> observe result -> think -> answer.

WHY THE DOCSTRING IS THE MOST IMPORTANT PART OF A TOOL
The docstring becomes the tool's description, and the description is the ONLY
thing the model uses to decide whether to call it. A vague docstring produces a
model that calls the wrong tool or no tool at all. Treat the docstring as a
prompt, because that is exactly what it is. Say what the tool does, when to use
it, and what it needs.

WHY PYDANTIC ARGUMENT SCHEMAS?
`args_schema` gives each parameter a type and a description the model can read,
and it VALIDATES the model's output before your function runs. If the model
hallucinates `order_id=12345` (an int), Pydantic rejects it and the agent gets a
correctable error message instead of your code raising a TypeError.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from app.database.database import fetch_customer_orders, fetch_order

logger = logging.getLogger(__name__)

# Order IDs look like ORD1001. Validating the shape lets us reject a
# hallucinated ID before we hit the database, and lets us give the model a
# precise correction message it can act on.
ORDER_ID_PATTERN = re.compile(r"^ORD\d{4}$")


class OrderStatusInput(BaseModel):
    """Arguments for get_order_status."""

    order_id: str = Field(
        description="The customer's order ID. Format: ORD followed by four "
                    "digits, for example ORD1001."
    )

    @field_validator("order_id")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()


class CustomerOrdersInput(BaseModel):
    """Arguments for get_customer_orders."""

    customer_id: str = Field(
        description="The customer ID. Format: CUST followed by three digits, "
                    "for example CUST001."
    )

    @field_validator("customer_id")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()


def _get_order_status(order_id: str) -> str:
    """Implementation behind the get_order_status tool.

    RETURNS A STRING, ALWAYS. Tool return values are fed back to the LLM as text,
    so every path -- success, not found, bad format -- must produce a string the
    model can reason about. Raising an exception would abort the agent run and
    show the customer a stack trace.

    We return JSON rather than a prose sentence because JSON is unambiguous:
    the model can see exactly which fields exist and which are null, and it is
    far less likely to misread a null as a value it can invent.
    """
    order_id = order_id.strip().upper()

    if not ORDER_ID_PATTERN.match(order_id):
        logger.warning("Rejected malformed order id: %r", order_id)
        return json.dumps({
            "found": False,
            "error": "invalid_format",
            "message": f"'{order_id}' is not a valid order ID. Order IDs look "
                       f"like ORD1001. Ask the customer to check their "
                       f"confirmation email.",
        })

    order = fetch_order(order_id)

    if order is None:
        logger.info("Order not found: %s", order_id)
        return json.dumps({
            "found": False,
            "error": "not_found",
            "message": f"No order with ID {order_id} exists in the system. Do "
                       f"not invent details. Ask the customer to double-check "
                       f"the ID, and escalate if they insist it is correct.",
        })

    logger.info("Order lookup succeeded: %s (%s)", order_id, order["status"])
    return json.dumps({"found": True, "order": order})


def _get_customer_orders(customer_id: str) -> str:
    """Implementation behind the get_customer_orders tool."""
    customer_id = customer_id.strip().upper()
    orders = fetch_customer_orders(customer_id)

    if not orders:
        return json.dumps({
            "found": False,
            "error": "not_found",
            "message": f"No orders found for customer {customer_id}. The "
                       f"customer ID may be wrong.",
        })

    return json.dumps({"found": True, "count": len(orders), "orders": orders})


# ---------------------------------------------------------------------------
# TOOL DEFINITIONS
#
# We use `StructuredTool.from_function` rather than the `@tool` decorator because
# it keeps the plain Python function importable and unit-testable on its own,
# while the tool object is what we hand to the agent. Testing `_get_order_status`
# directly is much simpler than testing through the tool wrapper.
# ---------------------------------------------------------------------------

get_order_status_tool = StructuredTool.from_function(
    func=_get_order_status,
    name="get_order_status",
    description=(
        "Look up the current status, expected delivery date, product, and total "
        "of a specific order. Use this whenever the customer asks where their "
        "order is, whether it has shipped, when it will arrive, what they "
        "ordered, or anything else about one specific order. Requires the order "
        "ID (format ORD1001). If the customer has not given an order ID, ask "
        "them for it instead of calling this tool with a guess."
    ),
    args_schema=OrderStatusInput,
    # If the tool raises anyway, hand the message back to the model as an
    # observation instead of crashing the run. The model can then apologise or
    # try a different approach.
    handle_tool_error=True,
)

get_customer_orders_tool = StructuredTool.from_function(
    func=_get_customer_orders,
    name="get_customer_orders",
    description=(
        "List every order placed by a specific customer, newest first. Use this "
        "when the customer asks about their order history or says something like "
        "'show me all my orders'. Requires the customer ID (format CUST001). Do "
        "NOT use this to look up a single order -- use get_order_status for that."
    ),
    args_schema=CustomerOrdersInput,
    handle_tool_error=True,
)

ORDER_TOOLS = [get_order_status_tool, get_customer_orders_tool]
