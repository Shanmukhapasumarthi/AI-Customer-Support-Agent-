from __future__ import annotations
from datetime import date
from decimal import Decimal
from sqlalchemy import Date, ForeignKey, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models. SQLAlchemy collects table definitions here,
    which is how `Base.metadata.create_all()` knows what to create."""


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    # Membership affects the return window (45 days instead of 30), so the
    # refund tool needs it.
    membership: Mapped[str] = mapped_column(String(20), default="standard")

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")

    def __repr__(self) -> str:
        return f"<Customer {self.customer_id} {self.name}>"


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    product_name: Mapped[str] = mapped_column(String(120), nullable=False)
    # Numeric, never float. Binary floats cannot represent 0.1 exactly, so
    # float money accumulates rounding errors. Numeric(10, 2) is exact decimal.
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    stock_quantity: Mapped[int] = mapped_column(default=0)
    warranty_months: Mapped[int] = mapped_column(default=12)

    orders: Mapped[list["Order"]] = relationship(back_populates="product")

    def __repr__(self) -> str:
        return f"<Product {self.product_id} {self.product_name}>"


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    # ForeignKey enforces that every order points at a real customer. Without
    # it, a typo creates an orphan row and the join silently returns nothing.
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    product_id: Mapped[str] = mapped_column(ForeignKey("products.product_id"))
    quantity: Mapped[int] = mapped_column(default=1)
    order_date: Mapped[date] = mapped_column(Date, nullable=False)
    # One of: Processing, Shipped, Out for Delivery, Delivered, Delayed,
    # Cancelled, Returned. These strings match the Shipping Policy document
    # exactly, so the agent can combine a tool result with a policy answer
    # without translating vocabulary.
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    expected_delivery: Mapped[date | None] = mapped_column(Date, nullable=True)
    delivered_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    shipping_method: Mapped[str] = mapped_column(String(30), default="Standard Shipping")

    customer: Mapped["Customer"] = relationship(back_populates="orders")
    product: Mapped["Product"] = relationship(back_populates="orders")

    def __repr__(self) -> str:
        return f"<Order {self.order_id} {self.status}>"
