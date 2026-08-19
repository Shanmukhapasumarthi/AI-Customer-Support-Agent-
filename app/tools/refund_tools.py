"""
PHASE 10 - LANGCHAIN TOOLS (REFUNDS AND CANCELLATION)

WHY CALCULATE THE REFUND IN PYTHON INSTEAD OF LETTING THE LLM DO IT?
Because LLMs are unreliable at arithmetic and at date maths, and this is money.
Asking a model to work out "delivered 20 days ago, 30-day window, 20% restocking
fee on $119.00" invites a wrong number stated with total confidence.

The division of labour that makes agents trustworthy:

    DETERMINISTIC LOGIC  ->  Python.   Dates, money, policy thresholds, eligibility.
    LANGUAGE             ->  the LLM.  Understanding the question, writing the reply.

The tool returns the decision AND the reasoning as data. The model's only job is
to phrase it. If you remember one design principle from this project, make it
this one: never let the model compute something you can compute exactly.

NOTE: this tool ESTIMATES eligibility. It never moves money. Actually issuing a
refund is a human action -- see the escalation policy. The tool description says
so explicitly, so the model does not promise a refund it cannot deliver.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from app.database.database import fetch_customer, fetch_order, today

logger = logging.getLogger(__name__)

# These constants encode the Refund Policy document. Keeping them here, named,
# means the policy is auditable in one place instead of being buried in prose
# inside a prompt where nobody can test it.
STANDARD_REFUND_WINDOW_DAYS = 30
PLUS_MEMBER_REFUND_WINDOW_DAYS = 45     # NimbusCart Plus perk, see faq.md
STRICT_INSPECTION_AFTER_DAYS = 15       # stricter condition check past this point
RESTOCKING_FEE_RATE = Decimal("0.20")   # 20%, applied to damaged packaging

# Statuses from which an order can still be cancelled free of charge.
CANCELLABLE_STATUSES = {"Processing", "Delayed"}
# Statuses that mean the item was never delivered, so no refund window applies.
NON_REFUNDABLE_STATUSES = {"Cancelled", "Returned"}


class RefundInput(BaseModel):
    order_id: str = Field(
        description="The order ID to estimate a refund for, format ORD1001."
    )
    packaging_intact: bool = Field(
        default=True,
        description="Whether the customer still has the original packaging and "
                    "all accessories. Set to false only if the customer has "
                    "explicitly said the packaging is damaged or missing. "
                    "Defaults to true.",
    )

    @field_validator("order_id")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()


class CancellationInput(BaseModel):
    order_id: str = Field(description="The order ID to check, format ORD1001.")

    @field_validator("order_id")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()


def _money(value: Decimal) -> str:
    """Round to cents using banker's-safe HALF_UP and return a string."""
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _calculate_refund_amount(order_id: str, packaging_intact: bool = True) -> str:
    """Implementation behind calculate_refund_amount."""
    order = fetch_order(order_id)

    if order is None:
        return json.dumps({
            "eligible": False,
            "error": "not_found",
            "message": f"No order {order_id} exists. Do not estimate a refund.",
        })

    status = order["status"]

    if status in NON_REFUNDABLE_STATUSES:
        return json.dumps({
            "eligible": False,
            "reason": f"Order {order_id} has status '{status}', so there is "
                      f"nothing to refund.",
        })

    if order["delivered_date"] is None:
        return json.dumps({
            "eligible": False,
            "reason": f"Order {order_id} has not been delivered yet (status: "
                      f"{status}). The 30-day refund window starts on the "
                      f"delivery date, so it has not begun. The customer can "
                      f"still cancel if the order has not shipped.",
            "order_status": status,
        })

    delivered = date.fromisoformat(order["delivered_date"])
    days_since_delivery = (today() - delivered).days

    # Plus members get a 45-day window. Look the membership up rather than
    # assuming, because getting this wrong in the customer's favour costs money
    # and getting it wrong the other way costs goodwill.
    customer = fetch_customer(order["customer_id"])
    is_plus = bool(customer and customer["membership"] == "plus")
    window = PLUS_MEMBER_REFUND_WINDOW_DAYS if is_plus else STANDARD_REFUND_WINDOW_DAYS

    if days_since_delivery > window:
        return json.dumps({
            "eligible": False,
            "reason": f"Order {order_id} was delivered {days_since_delivery} "
                      f"days ago, which is outside the {window}-day refund "
                      f"window. A refund cannot be offered.",
            "days_since_delivery": days_since_delivery,
            "refund_window_days": window,
            "membership": "plus" if is_plus else "standard",
        })

    total = Decimal(order["total_amount"])
    fee = total * RESTOCKING_FEE_RATE if not packaging_intact else Decimal("0.00")
    refund = total - fee

    result = {
        "eligible": True,
        "order_id": order_id,
        "order_total": _money(total),
        "restocking_fee": _money(fee),
        "estimated_refund": _money(refund),
        "days_since_delivery": days_since_delivery,
        "days_remaining_in_window": window - days_since_delivery,
        "refund_window_days": window,
        "membership": "plus" if is_plus else "standard",
        "strict_inspection": days_since_delivery > STRICT_INSPECTION_AFTER_DAYS,
        "processing_time": "2 business days for inspection, then 5-7 business "
                           "days for the money to reach the original payment method",
        "note": "This is an ESTIMATE based on policy. The final amount is "
                "confirmed after warehouse inspection. Do not promise this "
                "exact amount as final.",
    }

    logger.info("Refund estimate for %s: %s (eligible)", order_id, result["estimated_refund"])
    return json.dumps(result)


def _check_cancellation_eligibility(order_id: str) -> str:
    """Implementation behind check_cancellation_eligibility."""
    order = fetch_order(order_id)

    if order is None:
        return json.dumps({
            "cancellable": False,
            "error": "not_found",
            "message": f"No order {order_id} exists.",
        })

    status = order["status"]

    if status == "Cancelled":
        return json.dumps({
            "cancellable": False,
            "reason": f"Order {order_id} is already cancelled.",
            "order_status": status,
        })

    if status in CANCELLABLE_STATUSES:
        return json.dumps({
            "cancellable": True,
            "order_id": order_id,
            "order_status": status,
            "refund_time": "3 to 5 business days for card and wallet payments",
            "note": "The order CAN be cancelled free of charge. Tell the "
                    "customer a support specialist will process the "
                    "cancellation. Do not tell them it is already done.",
        })

    return json.dumps({
        "cancellable": False,
        "order_id": order_id,
        "order_status": status,
        "reason": f"Order {order_id} has status '{status}'. Cancellation is "
                  f"only possible while an order is Processing or Delayed. "
                  f"Once it has shipped it must go through the returns process "
                  f"instead.",
        "alternative": "The customer can refuse delivery at the door, or accept "
                       "it and return it within the 30-day window.",
    })


calculate_refund_amount_tool = StructuredTool.from_function(
    func=_calculate_refund_amount,
    name="calculate_refund_amount",
    description=(
        "Work out whether an order is still inside its refund window and how "
        "much money the customer would get back, including any restocking fee. "
        "Use this whenever a customer asks 'can I get a refund', 'how much will "
        "I get back', or 'am I still in time to return this'. Requires the "
        "order ID. This ESTIMATES eligibility only -- it does not issue a "
        "refund, so never tell the customer the refund has been processed."
    ),
    args_schema=RefundInput,
    handle_tool_error=True,
)

check_cancellation_tool = StructuredTool.from_function(
    func=_check_cancellation_eligibility,
    name="check_cancellation_eligibility",
    description=(
        "Check whether a specific order can still be cancelled, based on its "
        "current status. Use this when the customer says they want to cancel an "
        "order. Requires the order ID. This CHECKS eligibility only -- it does "
        "not perform the cancellation, so never tell the customer their order "
        "has been cancelled."
    ),
    args_schema=CancellationInput,
    handle_tool_error=True,
)

REFUND_TOOLS = [calculate_refund_amount_tool, check_cancellation_tool]
