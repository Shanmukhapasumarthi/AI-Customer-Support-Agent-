from __future__ import annotations

import json

import pytest

from app.tools import ALL_TOOLS, TOOL_NAMES
from app.tools.order_tools import _get_customer_orders, _get_order_status
from app.tools.product_tools import _get_product_information, _search_products
from app.tools.refund_tools import _calculate_refund_amount, _check_cancellation_eligibility


# --- Tool registry ---------------------------------------------------------

def test_all_tools_registered():
    assert set(TOOL_NAMES) == {
        "search_knowledge_base", "get_order_status", "get_customer_orders",
        "get_product_information", "search_products",
        "calculate_refund_amount", "check_cancellation_eligibility",
    }


def test_every_tool_has_a_substantial_description():
    """The description is the ONLY thing the model uses to pick a tool. A thin
    one produces wrong tool selection, so we enforce a floor."""
    for tool in ALL_TOOLS:
        assert len(tool.description) > 100, f"{tool.name} description is too thin"


def test_every_tool_has_an_args_schema():
    for tool in ALL_TOOLS:
        assert tool.args_schema is not None, f"{tool.name} has no args_schema"


# --- Order tools -----------------------------------------------------------

def test_order_lookup_succeeds(seeded_database):
    result = json.loads(_get_order_status("ORD1001"))
    assert result["found"] is True
    assert result["order"]["order_id"] == "ORD1001"
    assert result["order"]["status"] == "Shipped"
    assert result["order"]["product_name"]


@pytest.mark.parametrize("raw", ["ord1001", " ORD1001 ", "Ord1001"])
def test_order_lookup_normalises_input(seeded_database, raw):
    """Customers type lowercase and add spaces. Normalising in the tool means
    the prompt does not have to tell the model to do it."""
    assert json.loads(_get_order_status(raw))["found"] is True


def test_order_not_found_returns_structured_refusal(seeded_database):
    """The critical anti-hallucination behaviour: 'not found' must be DATA the
    model can read, with an explicit instruction not to invent."""
    result = json.loads(_get_order_status("ORD9999"))
    assert result["found"] is False
    assert result["error"] == "not_found"
    assert "do not invent" in result["message"].lower()


def test_malformed_order_id_rejected_before_database(seeded_database):
    result = json.loads(_get_order_status("my order"))
    assert result["found"] is False
    assert result["error"] == "invalid_format"


def test_order_lookup_never_raises(seeded_database):
    """Tool output is fed back to the LLM as text. An exception would abort the
    agent run and show the customer a stack trace."""
    for bad in ["", "   ", "'; DROP TABLE orders; --", "ORD" * 50, "12345"]:
        assert isinstance(_get_order_status(bad), str)


def test_customer_orders_returns_all(seeded_database):
    result = json.loads(_get_customer_orders("CUST001"))
    assert result["found"] is True
    assert result["count"] == 2


def test_customer_orders_sorted_newest_first(seeded_database):
    orders = json.loads(_get_customer_orders("CUST001"))["orders"]
    dates = [o["order_date"] for o in orders]
    assert dates == sorted(dates, reverse=True)


def test_unknown_customer_returns_not_found(seeded_database):
    assert json.loads(_get_customer_orders("CUST999"))["found"] is False


def test_order_output_excludes_customer_email(seeded_database):
    """Tool output goes into the model's context and can be quoted back. The
    tool boundary is a privacy boundary."""
    raw = _get_order_status("ORD1001")
    assert "@example.com" not in raw


# --- Product tools ---------------------------------------------------------

def test_product_lookup_returns_current_price(seeded_database):
    result = json.loads(_get_product_information("PROD001"))
    assert result["found"] is True
    assert result["product"]["price"] == "199.99"
    assert result["product"]["in_stock"] is True


def test_out_of_stock_product_flagged(seeded_database):
    result = json.loads(_get_product_information("PROD005"))
    assert result["product"]["in_stock"] is False


def test_product_search_by_partial_name(seeded_database):
    result = json.loads(_search_products("AuraSound"))
    assert result["found"] is True
    assert result["count"] == 2


def test_product_search_is_case_insensitive(seeded_database):
    assert json.loads(_search_products("aurasound"))["count"] == 2


def test_product_search_no_match_refuses_to_guess(seeded_database):
    result = json.loads(_search_products("quantum flux capacitor"))
    assert result["found"] is False
    assert "guessing" in result["message"].lower()


# --- Refund calculation (the money-critical logic) -------------------------

def test_refund_inside_window_is_full(seeded_database):
    """ORD1002: delivered 5 days ago, packaging intact -> full refund."""
    result = json.loads(_calculate_refund_amount("ORD1002", packaging_intact=True))
    assert result["eligible"] is True
    assert result["estimated_refund"] == "79.98"
    assert result["restocking_fee"] == "0.00"


def test_restocking_fee_applied_when_packaging_damaged(seeded_database):
    """20% of 79.98 = 15.996 -> 16.00 after HALF_UP rounding."""
    result = json.loads(_calculate_refund_amount("ORD1002", packaging_intact=False))
    assert result["restocking_fee"] == "16.00"
    assert result["estimated_refund"] == "63.98"


def test_refund_outside_window_refused(seeded_database):
    """ORD1003: delivered 40 days ago to a STANDARD member (30-day window)."""
    result = json.loads(_calculate_refund_amount("ORD1003"))
    assert result["eligible"] is False
    assert result["refund_window_days"] == 30
    assert result["days_since_delivery"] == 40


def test_plus_member_gets_longer_window(seeded_database):
    """ORD1002 belongs to CUST001, a Plus member -> 45-day window."""
    result = json.loads(_calculate_refund_amount("ORD1002"))
    assert result["membership"] == "plus"
    assert result["refund_window_days"] == 45


def test_undelivered_order_has_no_refund_window(seeded_database):
    """ORD1001 is still Shipped. The window starts at DELIVERY, so it has not
    begun -- a subtle rule the LLM would likely get wrong on its own."""
    result = json.loads(_calculate_refund_amount("ORD1001"))
    assert result["eligible"] is False
    assert "not been delivered" in result["reason"]


def test_strict_inspection_flag_after_15_days(seeded_database):
    """ORD1006: delivered 20 days ago -> inside window but past day 15."""
    result = json.loads(_calculate_refund_amount("ORD1006"))
    assert result["eligible"] is True
    assert result["strict_inspection"] is True


def test_recent_delivery_not_flagged_strict(seeded_database):
    result = json.loads(_calculate_refund_amount("ORD1002"))
    assert result["strict_inspection"] is False


def test_cancelled_order_has_nothing_to_refund(seeded_database):
    result = json.loads(_calculate_refund_amount("ORD1007"))
    assert result["eligible"] is False


def test_refund_result_warns_against_promising_finality(seeded_database):
    """The tool must not let the agent promise an exact final amount."""
    result = json.loads(_calculate_refund_amount("ORD1002"))
    assert "estimate" in result["note"].lower()


def test_refund_for_unknown_order(seeded_database):
    result = json.loads(_calculate_refund_amount("ORD9999"))
    assert result["eligible"] is False
    assert result["error"] == "not_found"


# --- Cancellation ----------------------------------------------------------

def test_processing_order_is_cancellable(seeded_database):
    result = json.loads(_check_cancellation_eligibility("ORD1004"))
    assert result["cancellable"] is True


def test_delayed_order_is_cancellable(seeded_database):
    """Per the Cancellation Policy, Delayed is treated like Processing."""
    assert json.loads(_check_cancellation_eligibility("ORD1005"))["cancellable"] is True


def test_shipped_order_is_not_cancellable(seeded_database):
    result = json.loads(_check_cancellation_eligibility("ORD1001"))
    assert result["cancellable"] is False
    assert "returns process" in result["reason"]
    assert "alternative" in result


def test_out_for_delivery_is_not_cancellable(seeded_database):
    assert json.loads(_check_cancellation_eligibility("ORD1008"))["cancellable"] is False


def test_already_cancelled_order(seeded_database):
    result = json.loads(_check_cancellation_eligibility("ORD1007"))
    assert result["cancellable"] is False
    assert "already cancelled" in result["reason"]


def test_cancellation_result_forbids_claiming_completion(seeded_database):
    """The agent must never tell a customer their order HAS been cancelled."""
    result = json.loads(_check_cancellation_eligibility("ORD1004"))
    assert "do not tell them it is already done" in result["note"].lower()


# --- Database layer --------------------------------------------------------

def test_database_seeded_correctly(seeded_database):
    from app.database.database import database_stats

    stats = database_stats()
    assert stats == {"customers": 4, "products": 8, "orders": 8}


def test_money_stored_exactly_not_as_float(seeded_database):
    """Numeric(10,2), not float. 0.1 has no exact binary representation, so
    float money accumulates rounding errors."""
    from decimal import Decimal

    from app.database.database import fetch_product

    price = fetch_product("PROD003")["price"]
    assert Decimal(price) == Decimal("149.50")
