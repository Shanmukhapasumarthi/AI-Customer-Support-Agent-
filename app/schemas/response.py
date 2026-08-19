from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class Intent(str, Enum):
    """What the customer is trying to do.

    A closed set, not a free-text string. Closed sets are what make analytics
    possible ("escalation_dispute rose 30% this week") and stop the model from
    inventing a new label on every request.

    Inheriting from `str` as well as `Enum` means these serialise to plain JSON
    strings automatically.
    """

    ORDER_STATUS = "order_status"
    ORDER_CANCELLATION = "order_cancellation"
    REFUND_POLICY = "refund_policy"
    REFUND_REQUEST = "refund_request"
    RETURN_POLICY = "return_policy"
    SHIPPING_POLICY = "shipping_policy"
    WARRANTY_POLICY = "warranty_policy"
    PAYMENT_POLICY = "payment_policy"
    PRODUCT_INFO = "product_info"
    PRODUCT_COMPARISON = "product_comparison"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class AnswerSource(str, Enum):
    """Where the information in the answer actually came from."""

    KNOWLEDGE_BASE = "knowledge_base"   # retrieved policy documents
    ORDER_DATABASE = "order_database"   # a database tool
    BOTH = "both"                       # combined KB + tool
    NONE = "none"                       # no grounding: greeting, or "I don't know"


class SupportResponse(BaseModel):
    """The contract every customer-facing answer must satisfy."""

    answer: str = Field(
        description="The natural-language reply shown to the customer."
    )
    intent: Intent = Field(
        default=Intent.UNKNOWN,
        description="Classified customer intent.",
    )
    source: AnswerSource = Field(
        default=AnswerSource.NONE,
        description="Where the information came from. Set from OBSERVED tool "
                    "usage, not from the model's own claim.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the system is in this answer, 0.0 to 1.0.",
    )
    requires_human: bool = Field(
        default=False,
        description="True when a human specialist must take over.",
    )
    escalation_reason: str | None = Field(
        default=None,
        description="One short sentence explaining why escalation is needed. "
                    "Null when requires_human is false.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Knowledge-base filenames that backed this answer.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of tools the agent actually called.",
    )
    session_id: str | None = Field(
        default=None, description="Conversation this reply belongs to."
    )
    latency_ms: int | None = Field(
        default=None, description="End-to-end processing time in milliseconds."
    )

    @model_validator(mode="after")
    def reason_required_when_escalating(self) -> "SupportResponse":
        """Enforce the invariant: escalating without a reason is a bug.

        A human picking this case out of a queue needs to know why it arrived
        there. Silently allowing a null reason would let that break unnoticed.

        WHY model_validator AND NOT field_validator?
        A `field_validator` on `escalation_reason` does NOT run when the field
        is left at its default of None -- Pydantic skips validation of defaults
        unless you set validate_default=True. That is exactly the case we care
        about here, so the field validator would never fire. A
        `model_validator(mode="after")` always runs, once, with every field
        already populated. (Our test suite caught this; it is a genuinely easy
        mistake to make.)
        """
        if self.requires_human and not self.escalation_reason:
            self.escalation_reason = "Escalated to a human specialist for review."
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Your order ORD1001 has shipped and is expected to "
                              "arrive on 21 August.",
                    "intent": "order_status",
                    "source": "order_database",
                    "confidence": 0.95,
                    "requires_human": False,
                    "escalation_reason": None,
                    "sources": [],
                    "tools_used": ["get_order_status"],
                }
            ]
        }
    }


class ResponseClassification(BaseModel):
    """The narrow slice the classifier LLM is asked to produce.

    Deliberately SMALLER than SupportResponse. We only ask the model for the
    things that genuinely require judgement -- intent, confidence, whether a
    human is needed. Everything factual (which tools ran, which files were read)
    is filled in from observation. Asking a model for less means it gets more of
    it right.
    """

    intent: Intent = Field(description="The customer's intent.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that the answer is correct and complete. Use "
                    "0.9+ when the answer came from a tool or a clear policy "
                    "quote, 0.4-0.7 when partially answered, below 0.3 when the "
                    "assistant said it does not know.",
    )
    requires_human: bool = Field(
        description="True if this case must go to a human specialist."
    )
    escalation_reason: str | None = Field(
        default=None,
        description="One short sentence on why a human is needed, or null.",
    )


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(min_length=1, max_length=2000,
                         description="The customer's message.")
    session_id: str = Field(default="default",
                            description="Conversation ID. Same ID = same memory.")

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("message cannot be empty or whitespace only")
        return cleaned


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: str
    app_name: str
    llm_model: str
    vector_store_ready: bool
    database_ready: bool
    chunks_indexed: int | None = None
    database_rows: dict[str, int] | None = None
    detail: str | None = None
