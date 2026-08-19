"""Tool registry.

One place that assembles every tool the agent can use. The agent imports
`ALL_TOOLS` and nothing else, so adding a capability is a one-line change here
rather than an edit inside the agent.

ORDER MATTERS SLIGHTLY: models pay marginally more attention to tools listed
first. We lead with the knowledge base because policy questions are the most
common category in customer support.
"""

from app.tools.knowledge_tools import KNOWLEDGE_TOOLS
from app.tools.order_tools import ORDER_TOOLS
from app.tools.product_tools import PRODUCT_TOOLS
from app.tools.refund_tools import REFUND_TOOLS

ALL_TOOLS = [*KNOWLEDGE_TOOLS, *ORDER_TOOLS, *PRODUCT_TOOLS, *REFUND_TOOLS]

TOOL_NAMES = [t.name for t in ALL_TOOLS]

__all__ = ["ALL_TOOLS", "TOOL_NAMES", "KNOWLEDGE_TOOLS", "ORDER_TOOLS",
           "PRODUCT_TOOLS", "REFUND_TOOLS"]
