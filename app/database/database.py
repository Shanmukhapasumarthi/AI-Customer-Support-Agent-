from __future__ import annotations
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database.models import Base, Customer, Order, Product

logger = logging.getLogger(__name__)

# `future=True` opts into SQLAlchemy 2.0 behaviour.
# `check_same_thread=False` is required because FastAPI serves requests from a
# thread pool, and SQLite otherwise refuses to be used across threads.
engine = create_engine(
    f"sqlite:///{settings.sqlite_path}",
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_database() -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    logger.info("Database ready at %s", settings.sqlite_path)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a session and guarantee it is closed, rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Serialisation helpers: ORM object -> plain dict
#
# These are also where we control WHAT THE LLM SEES. Note that we never expose
# the customer's email through an order lookup. Tool output goes straight into
# the model's context and can end up quoted back to whoever is chatting, so the
# tool boundary is a privacy boundary. Only include fields the answer needs.
# ---------------------------------------------------------------------------

def _order_to_dict(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "customer_name": order.customer.name if order.customer else None,
        "product_id": order.product_id,
        "product_name": order.product.product_name if order.product else None,
        "quantity": order.quantity,
        "order_date": order.order_date.isoformat(),
        "status": order.status,
        "expected_delivery": order.expected_delivery.isoformat()
        if order.expected_delivery
        else None,
        "delivered_date": order.delivered_date.isoformat() if order.delivered_date else None,
        # str() on Decimal keeps exact cents. float() would reintroduce the
        # rounding error we chose Numeric to avoid.
        "total_amount": str(order.total_amount),
        "shipping_method": order.shipping_method,
    }


def _product_to_dict(product: Product) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "product_name": product.product_name,
        "price": str(product.price),
        "category": product.category,
        "in_stock": product.stock_quantity > 0,
        "stock_quantity": product.stock_quantity,
        "warranty_months": product.warranty_months,
    }


# ---------------------------------------------------------------------------
# Query functions used by the LangChain tools.
# Each returns None (not an exception) when nothing is found, so the tool layer
# can turn "not found" into a helpful sentence rather than a stack trace.
# ---------------------------------------------------------------------------

def fetch_order(order_id: str) -> dict[str, Any] | None:
    """Look up a single order by exact ID (case-insensitive)."""
    with get_session() as session:
        # .upper() + .strip() because customers type "ord1001" and " ORD1001 ".
        # Normalising here means the agent does not need to be told to do it.
        order = session.get(Order, order_id.strip().upper())
        return _order_to_dict(order) if order else None


def fetch_customer_orders(customer_id: str) -> list[dict[str, Any]]:
    """All orders for a customer, newest first."""
    with get_session() as session:
        stmt = (
            select(Order)
            .where(Order.customer_id == customer_id.strip().upper())
            .order_by(Order.order_date.desc())
        )
        return [_order_to_dict(o) for o in session.scalars(stmt).all()]


def fetch_product(product_id: str) -> dict[str, Any] | None:
    """Look up a product by exact ID."""
    with get_session() as session:
        product = session.get(Product, product_id.strip().upper())
        return _product_to_dict(product) if product else None


def search_products_by_name(name_fragment: str) -> list[dict[str, Any]]:
    """Find products whose name contains the given text (case-insensitive).

    Customers say "the AuraSound headphones", not "PROD001". This lets the agent
    resolve a name to an ID without guessing.
    """
    with get_session() as session:
        stmt = select(Product).where(Product.product_name.ilike(f"%{name_fragment.strip()}%"))
        return [_product_to_dict(p) for p in session.scalars(stmt).all()]


def fetch_customer(customer_id: str) -> dict[str, Any] | None:
    with get_session() as session:
        customer = session.get(Customer, customer_id.strip().upper())
        if not customer:
            return None
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "membership": customer.membership,
            "order_count": len(customer.orders),
        }


def database_stats() -> dict[str, int]:
    """Row counts. Used by the /health endpoint to prove the DB is seeded."""
    with get_session() as session:
        return {
            "customers": session.query(Customer).count(),
            "products": session.query(Product).count(),
            "orders": session.query(Order).count(),
        }


def today() -> date:
    """Single source of 'now' for refund-window maths.

    Isolated in one function so tests can monkeypatch it and get deterministic
    results instead of failing whenever the calendar moves.
    """
    return date.today()
