"""
The order set is deliberately built to exercise every branch of the agent:
  ORD1001  Shipped              -> "where is my order?"
  ORD1002  Delivered 5 days ago -> inside refund window, full refund
  ORD1003  Delivered 40 days ago-> outside refund window, refund refused
  ORD1004  Processing           -> cancellable for free
  ORD1005  Delayed              -> cancellable, and a delay explanation
  ORD1006  Delivered 20 days ago-> inside window but past day 15 (strict check)
  ORD1007  Cancelled            -> already cancelled
  ORD1008  Out for Delivery     -> arriving today, cannot cancel
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal

from app.config import setup_logging
from app.database.database import SessionLocal, engine, init_database
from app.database.models import Base, Customer, Order, Product

logger = logging.getLogger(__name__)

TODAY = date.today()


def d(days_offset: int) -> date:
    """Return a date relative to today. Negative = past, positive = future."""
    return TODAY + timedelta(days=days_offset)


CUSTOMERS = [
    Customer(customer_id="CUST001", name="Anita Rao", email="anita.rao@example.com",
             membership="plus"),
    Customer(customer_id="CUST002", name="Daniel Okafor", email="d.okafor@example.com",
             membership="standard"),
    Customer(customer_id="CUST003", name="Mei Lin Chen", email="meilin.chen@example.com",
             membership="standard"),
    Customer(customer_id="CUST004", name="Tomas Alvarez", email="t.alvarez@example.com",
             membership="plus"),
]

PRODUCTS = [
    Product(product_id="PROD001", product_name="AuraSound X1 Wireless Headphones",
            price=Decimal("199.99"), category="Audio", stock_quantity=42, warranty_months=12),
    Product(product_id="PROD002", product_name="AuraSound Mini True Wireless Earbuds",
            price=Decimal("89.99"), category="Audio", stock_quantity=118, warranty_months=12),
    Product(product_id="PROD003", product_name="PulseTrack Fit Smartwatch",
            price=Decimal("149.50"), category="Wearables",
            stock_quantity=27, warranty_months=12),
    Product(product_id="PROD004", product_name="BrewMaster Duo Coffee Machine",
            price=Decimal("329.00"), category="Home Appliances",
            stock_quantity=8, warranty_months=24),
    Product(product_id="PROD005", product_name="NimbusGlow Smart Desk Lamp",
            price=Decimal("59.99"), category="Home Appliances",
            stock_quantity=0, warranty_months=24),
    Product(product_id="PROD006", product_name="SwiftCharge 65W GaN Charger",
            price=Decimal("39.99"), category="Accessories",
            stock_quantity=203, warranty_months=6),
    Product(product_id="PROD007", product_name="TrailPack Pro 30L Travel Backpack",
            price=Decimal("119.00"), category="Accessories",
            stock_quantity=15, warranty_months=6),
    Product(product_id="PROD008", product_name="CloudStep Runner Running Shoes",
            price=Decimal("94.95"), category="Apparel",
            stock_quantity=61, warranty_months=3),
]

ORDERS = [
    Order(order_id="ORD1001", customer_id="CUST001", product_id="PROD001", quantity=1,
          order_date=d(-6), status="Shipped", expected_delivery=d(3),
          total_amount=Decimal("199.99"), shipping_method="Standard Shipping"),

    Order(order_id="ORD1002", customer_id="CUST001", product_id="PROD006", quantity=2,
          order_date=d(-12), status="Delivered", expected_delivery=d(-5),
          delivered_date=d(-5), total_amount=Decimal("79.98"),
          shipping_method="Express Shipping"),

    Order(order_id="ORD1003", customer_id="CUST002", product_id="PROD004", quantity=1,
          order_date=d(-48), status="Delivered", expected_delivery=d(-40),
          delivered_date=d(-40), total_amount=Decimal("329.00"),
          shipping_method="Standard Shipping"),

    Order(order_id="ORD1004", customer_id="CUST002", product_id="PROD002", quantity=1,
          order_date=d(-1), status="Processing", expected_delivery=d(7),
          total_amount=Decimal("89.99"), shipping_method="Standard Shipping"),

    Order(order_id="ORD1005", customer_id="CUST003", product_id="PROD008", quantity=1,
          order_date=d(-14), status="Delayed", expected_delivery=d(-2),
          total_amount=Decimal("94.95"), shipping_method="Standard Shipping"),

    Order(order_id="ORD1006", customer_id="CUST003", product_id="PROD007", quantity=1,
          order_date=d(-26), status="Delivered", expected_delivery=d(-20),
          delivered_date=d(-20), total_amount=Decimal("119.00"),
          shipping_method="Standard Shipping"),

    Order(order_id="ORD1007", customer_id="CUST004", product_id="PROD005", quantity=1,
          order_date=d(-9), status="Cancelled", expected_delivery=None,
          total_amount=Decimal("59.99"), shipping_method="Standard Shipping"),

    Order(order_id="ORD1008", customer_id="CUST004", product_id="PROD003", quantity=1,
          order_date=d(-4), status="Out for Delivery", expected_delivery=TODAY,
          total_amount=Decimal("149.50"), shipping_method="Express Shipping"),
]


def seed(reset: bool = True) -> None:
    """Create the schema and insert the fictional data.

    Args:
        reset: Drop existing tables first. Default True so re-running the script
            is idempotent instead of raising primary-key conflicts.
    """
    if reset:
        logger.info("Dropping existing tables...")
        Base.metadata.drop_all(engine)

    init_database()

    session = SessionLocal()
    try:
        # Order matters: customers and products must exist before orders, or the
        # foreign keys point at nothing.
        session.add_all(CUSTOMERS)
        session.add_all(PRODUCTS)
        session.flush()  # push to the DB without committing, so FKs resolve
        session.add_all(ORDERS)
        session.commit()
        logger.info(
            "Seeded %d customers, %d products, %d orders.",
            len(CUSTOMERS), len(PRODUCTS), len(ORDERS),
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    setup_logging()
    seed()

    from app.database.database import database_stats, fetch_order

    print("\nRow counts:", database_stats())
    print("\nSample orders:")
    for oid in ["ORD1001", "ORD1003", "ORD1004"]:
        o = fetch_order(oid)
        print(f"  {o['order_id']}  {o['status']:<18} {o['product_name'][:34]:<34} "
              f"${o['total_amount']}")
