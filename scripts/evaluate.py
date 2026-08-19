from __future__ import annotations
import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

from app.config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """One evaluation question and what we expect from it."""

    question: str
    category: str
    # Tools we expect to be called. Empty means "no tool needed".
    expected_tools: list[str] = field(default_factory=list)
    # Lowercased substrings that should appear in the answer.
    expected_keywords: list[str] = field(default_factory=list)
    # Should the agent hand off to a human?
    expect_escalation: bool = False
    # Should the agent REFUSE to answer? (unanswerable questions)
    expect_refusal: bool = False
    # Multi-turn cases share a session id and run in listed order.
    session_id: str = "eval"


EVAL_CASES: list[EvalCase] = [
    # ---- Knowledge base (6) ------------------------------------------------
    EvalCase(
        question="What is your refund policy?",
        category="knowledge_base",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["30"],
    ),
    EvalCase(
        question="How long does shipping take?",
        category="knowledge_base",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["business days"],
    ),
    EvalCase(
        question="Can I return an item after 15 days?",
        category="knowledge_base",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["30"],
    ),
    EvalCase(
        question="What is the warranty period?",
        category="knowledge_base",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["12"],
    ),
    EvalCase(
        question="Can I cancel my order after it has shipped?",
        category="knowledge_base",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["return"],
    ),
    EvalCase(
        question="What is the difference between the AuraSound X1 and the Mini?",
        category="knowledge_base",
        expected_tools=["search_knowledge_base"],
        expected_keywords=["battery"],
    ),

    # ---- Tools (6) ---------------------------------------------------------
    EvalCase(
        question="Where is order ORD1001?",
        category="tool",
        expected_tools=["get_order_status"],
        expected_keywords=["ord1001"],
    ),
    EvalCase(
        question="What did I order in ORD1002?",
        category="tool",
        expected_tools=["get_order_status"],
        expected_keywords=["swiftcharge"],
    ),
    EvalCase(
        question="What is the price of PROD001?",
        category="tool",
        expected_tools=["get_product_information"],
        expected_keywords=["199.99"],
    ),
    EvalCase(
        question="What is the refund amount for ORD1002?",
        category="tool",
        expected_tools=["calculate_refund_amount"],
        expected_keywords=["79.98"],
    ),
    EvalCase(
        question="I want to cancel order ORD1004.",
        category="tool",
        expected_tools=["check_cancellation_eligibility"],
        expected_keywords=["cancel"],
    ),
    EvalCase(
        question="Can I get a refund for ORD1003?",
        category="tool",
        expected_tools=["calculate_refund_amount"],
        expected_keywords=["30"],
    ),

    # ---- Multi-turn memory (3, run in order, same session) -----------------
    EvalCase(
        question="My order is ORD1001.",
        category="memory",
        session_id="eval-memory",
    ),
    EvalCase(
        question="When will it arrive?",
        category="memory",
        expected_tools=["get_order_status"],
        session_id="eval-memory",
    ),
    EvalCase(
        question="What product did I order?",
        category="memory",
        expected_keywords=["aurasound"],
        session_id="eval-memory",
    ),

    # ---- Unanswerable: hallucination check (4) -----------------------------
    EvalCase(
        question="Who is the CEO of NimbusCart?",
        category="unanswerable",
        expect_refusal=True,
    ),
    EvalCase(
        question="What is your head office address?",
        category="unanswerable",
        expect_refusal=True,
    ),
    EvalCase(
        question="Where is order ORD9999?",
        category="unanswerable",
        expected_tools=["get_order_status"],
        expect_refusal=True,
    ),
    EvalCase(
        question="How many employees does NimbusCart have?",
        category="unanswerable",
        expect_refusal=True,
    ),

    # ---- Escalation (3) ----------------------------------------------------
    EvalCase(
        question="I want to dispute my refund. The amount you gave me is wrong.",
        category="escalation",
        expect_escalation=True,
    ),
    EvalCase(
        question="My payment was charged twice and I need it fixed immediately.",
        category="escalation",
        expect_escalation=True,
    ),
    EvalCase(
        question="My product arrived damaged and I am extremely upset about it.",
        category="escalation",
        expect_escalation=True,
    ),
]

# Phrases that indicate a refusal. Kept in sync with UNCERTAINTY_MARKERS in
# support_agent.py, plus the "not found" wording the order tool triggers.
REFUSAL_MARKERS = (
    "don't have", "do not have", "not in my knowledge base", "couldn't find",
    "could not find", "no order", "unable to find", "don't know", "do not know",
    "not able to find", "doesn't exist", "does not exist",
)


@dataclass
class EvalResult:
    question: str
    category: str
    answer: str
    tools_used: list[str]
    expected_tools: list[str]
    intent: str
    source: str
    confidence: float
    requires_human: bool
    latency_ms: int
    tool_ok: bool
    keywords_ok: bool
    escalation_ok: bool
    refusal_ok: bool
    passed: bool


def run_case(case: EvalCase) -> EvalResult:
    from app.agents.support_agent import answer_question

    started = time.perf_counter()
    response = answer_question(case.question, session_id=case.session_id)
    latency = int((time.perf_counter() - started) * 1000)

    answer_lower = response.answer.lower()

    # Tool check: every expected tool must appear. Extra tools are allowed --
    # calling the knowledge base as well as an order tool is often correct.
    tool_ok = all(t in response.tools_used for t in case.expected_tools)

    keywords_ok = all(kw.lower() in answer_lower for kw in case.expected_keywords)
    escalation_ok = response.requires_human == case.expect_escalation

    refused = any(marker in answer_lower for marker in REFUSAL_MARKERS)
    # For unanswerable questions, refusing is success. For everything else,
    # refusing when we expected an answer is a failure.
    refusal_ok = refused if case.expect_refusal else True

    passed = tool_ok and keywords_ok and escalation_ok and refusal_ok

    return EvalResult(
        question=case.question,
        category=case.category,
        answer=response.answer,
        tools_used=response.tools_used,
        expected_tools=case.expected_tools,
        intent=response.intent.value,
        source=response.source.value,
        confidence=response.confidence,
        requires_human=response.requires_human,
        latency_ms=latency,
        tool_ok=tool_ok,
        keywords_ok=keywords_ok,
        escalation_ok=escalation_ok,
        refusal_ok=refusal_ok,
        passed=passed,
    )


def summarise(results: list[EvalResult]) -> dict:
    """Compute the headline metrics."""
    total = len(results)
    latencies = [r.latency_ms for r in results]

    with_tools = [r for r in results if r.expected_tools]
    unanswerable = [r for r in results if r.category == "unanswerable"]
    escalation = [r for r in results if r.category == "escalation"]

    def pct(numerator: int, denominator: int) -> float:
        return round(100 * numerator / denominator, 1) if denominator else 0.0

    by_category: dict[str, dict] = {}
    for r in results:
        entry = by_category.setdefault(r.category, {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += int(r.passed)
    for entry in by_category.values():
        entry["rate"] = pct(entry["passed"], entry["total"])

    return {
        "total_cases": total,
        "passed": sum(r.passed for r in results),
        "overall_pass_rate": pct(sum(r.passed for r in results), total),
        "tool_selection_accuracy": pct(
            sum(r.tool_ok for r in with_tools), len(with_tools)
        ),
        "keyword_recall": pct(
            sum(r.keywords_ok for r in results), total
        ),
        # A hallucination here means: we asked something unanswerable and the
        # agent answered anyway instead of refusing.
        "hallucination_rate": pct(
            sum(not r.refusal_ok for r in unanswerable), len(unanswerable)
        ),
        "escalation_accuracy": pct(
            sum(r.escalation_ok for r in escalation), len(escalation)
        ),
        "false_escalation_rate": pct(
            sum(r.requires_human for r in results if r.category in
                {"knowledge_base", "tool"}),
            len([r for r in results if r.category in {"knowledge_base", "tool"}]),
        ),
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": int(
            statistics.quantiles(latencies, n=20)[18]
        ) if len(latencies) >= 20 else max(latencies, default=0),
        "mean_confidence": round(
            statistics.mean(r.confidence for r in results), 2
        ) if results else 0.0,
        "by_category": by_category,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the support agent.")
    parser.add_argument("--category", help="Run only one category.")
    parser.add_argument("--json", help="Write full results to this JSON file.")
    args = parser.parse_args()

    setup_logging("WARNING")  # quiet: we want the report, not the agent's logs

    cases = EVAL_CASES
    if args.category:
        cases = [c for c in cases if c.category == args.category]
        if not cases:
            print(f"No cases in category {args.category!r}.")
            return 1

    print(f"\nRunning {len(cases)} evaluation cases...\n")
    print(f"{'':<3} {'CATEGORY':<14} {'QUESTION':<52} {'TOOLS':<7} {'MS':>6}")
    print("-" * 90)

    results: list[EvalResult] = []
    for i, case in enumerate(cases, 1):
        result = run_case(case)
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        print(f"{mark:<3} {result.category:<14} {case.question[:50]:<52} "
              f"{'ok' if result.tool_ok else 'WRONG':<7} {result.latency_ms:>6}")

    summary = summarise(results)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Overall pass rate        : {summary['overall_pass_rate']}%  "
          f"({summary['passed']}/{summary['total_cases']})")
    print(f"Tool selection accuracy  : {summary['tool_selection_accuracy']}%")
    print(f"Keyword recall           : {summary['keyword_recall']}%")
    print(f"Hallucination rate       : {summary['hallucination_rate']}%  (lower is better)")
    print(f"Escalation accuracy      : {summary['escalation_accuracy']}%")
    print(f"False escalation rate    : {summary['false_escalation_rate']}%  (lower is better)")
    print(f"Latency p50 / p95        : {summary['latency_p50_ms']} ms / "
          f"{summary['latency_p95_ms']} ms")
    print(f"Mean confidence          : {summary['mean_confidence']}")

    print("\nBy category:")
    for category, stats in sorted(summary["by_category"].items()):
        print(f"  {category:<16} {stats['passed']}/{stats['total']}  ({stats['rate']}%)")

    failures = [r for r in results if not r.passed]
    if failures:
        print("\n" + "=" * 90)
        print("FAILURES - read these, they tell you which layer to tune")
        print("=" * 90)
        for r in failures:
            print(f"\nQ: {r.question}")
            print(f"A: {r.answer[:200]}")
            reasons = []
            if not r.tool_ok:
                reasons.append(f"expected tools {r.expected_tools}, got {r.tools_used} "
                               "-> improve that tool's DESCRIPTION")
            if not r.keywords_ok:
                reasons.append("missing expected facts -> tune chunk size / top-k / threshold")
            if not r.escalation_ok:
                reasons.append(f"requires_human={r.requires_human} "
                               "-> tune CLASSIFIER_SYSTEM_PROMPT criteria")
            if not r.refusal_ok:
                reasons.append("HALLUCINATION: answered an unanswerable question "
                               "-> raise SIMILARITY_THRESHOLD")
            for reason in reasons:
                print(f"   - {reason}")

    if args.json:
        payload = {"summary": summary, "results": [asdict(r) for r in results]}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nFull results written to {args.json}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
