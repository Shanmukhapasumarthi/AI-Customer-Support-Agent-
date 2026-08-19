"""
PHASE 10 - LANGCHAIN TOOLS (PRODUCTS)

WHY PRODUCT TOOLS WHEN WE ALSO HAVE product_information.md IN THE KNOWLEDGE BASE?
This is the clearest illustration in the whole project of when to use retrieval
and when to use a tool.

  * The knowledge base holds STATIC product SPECIFICATIONS: battery life, water
    resistance, how the X1 differs from the Mini. That text changes rarely and is
    best answered by semantic search, because customers ask about it in endless
    different phrasings.

  * The database holds VOLATILE product FACTS: current price and stock level.
    These change daily. If you embed a price into the vector store, the moment
    the price changes your agent starts confidently quoting a stale number, and
    you will not notice until a customer complains.

RULE OF THUMB: if a fact would be wrong tomorrow, it belongs in a tool, not in a
vector store. The agent decides which to use based on the tool descriptions, and
"current price and stock" appears prominently in the description below for
exactly that reason.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from app.database.database import fetch_product, search_products_by_name

logger = logging.getLogger(__name__)

PRODUCT_ID_PATTERN = re.compile(r"^PROD\d{3}$")


class ProductInput(BaseModel):
    product_id: str = Field(
        description="The product ID. Format: PROD followed by three digits, "
                    "for example PROD001."
    )

    @field_validator("product_id")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()


class ProductSearchInput(BaseModel):
    name_fragment: str = Field(
        description="Part of the product name to search for, for example "
                    "'AuraSound' or 'backpack'. Case-insensitive."
    )


def _get_product_information(product_id: str) -> str:
    """Implementation behind get_product_information."""
    product_id = product_id.strip().upper()

    if not PRODUCT_ID_PATTERN.match(product_id):
        return json.dumps({
            "found": False,
            "error": "invalid_format",
            "message": f"'{product_id}' is not a valid product ID. Product IDs "
                       f"look like PROD001. If the customer gave a product NAME "
                       f"instead, use search_products by name.",
        })

    product = fetch_product(product_id)

    if product is None:
        return json.dumps({
            "found": False,
            "error": "not_found",
            "message": f"No product with ID {product_id} exists. Do not invent "
                       f"a price or specification.",
        })

    logger.info("Product lookup succeeded: %s", product_id)
    return json.dumps({"found": True, "product": product})


def _search_products(name_fragment: str) -> str:
    """Implementation behind search_products."""
    matches = search_products_by_name(name_fragment)

    if not matches:
        return json.dumps({
            "found": False,
            "error": "not_found",
            "message": f"No products matched '{name_fragment}'. Tell the "
                       f"customer we do not appear to carry that item rather "
                       f"than guessing at a similar product.",
        })

    logger.info("Product search %r matched %d products", name_fragment, len(matches))
    return json.dumps({"found": True, "count": len(matches), "products": matches})


get_product_information_tool = StructuredTool.from_function(
    func=_get_product_information,
    name="get_product_information",
    description=(
        "Get the CURRENT price, stock availability, category, and warranty "
        "length of a product by its product ID (format PROD001). Use this "
        "whenever the customer asks how much something costs, whether it is in "
        "stock, or how long its warranty is. Always use this tool for price and "
        "stock -- never state a price from memory or from policy documents, "
        "because prices change."
    ),
    args_schema=ProductInput,
    handle_tool_error=True,
)

search_products_tool = StructuredTool.from_function(
    func=_search_products,
    name="search_products",
    description=(
        "Find products by name or partial name when the customer does not know "
        "the product ID, for example 'the AuraSound headphones' or 'backpack'. "
        "Returns matching products with their IDs and current prices. Use this "
        "first when the customer names a product instead of giving an ID."
    ),
    args_schema=ProductSearchInput,
    handle_tool_error=True,
)

PRODUCT_TOOLS = [get_product_information_tool, search_products_tool]
