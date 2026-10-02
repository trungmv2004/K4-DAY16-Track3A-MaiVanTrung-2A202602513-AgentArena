"""Behaviour checks for evidence, tool budgets and the five student layers."""

from types import SimpleNamespace

import pytest

from arena.corpus import Corpus, Doc, INJECTION_CANARY
from arena.model import DEGRADED_MARKERS, FINALIZE_SENTINEL
from arena.tools import ToolResult
from harness.agent import AgentContext
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, PLACEHOLDER, InjectionGuard
from harness.layers.retry import Retry


def context(docs=(), observations=(), budget=8, calls=0):
    return AgentContext(
        brief={"budget": {"max_tool_calls": budget}},
        tools=SimpleNamespace(calls=calls), trace=None,
        corpus=Corpus(list(docs)), observations=list(observations),
    )


def doc(doc_id, body):
    return Doc(doc_id=doc_id, title=doc_id, body=body, tags=())


@pytest.mark.parametrize("budget,calls,spent", [(8, 6, False), (8, 7, True),
                                              (None, 100, False), (1, 0, True)])
def test_budget_nudge_preserves_history_and_question(budget, calls, spent):
    ctx = context(budget=budget, calls=calls)
    history = [{"role": "user", "content": "Câu hỏi gốc"}]
    outbound = BudgetPolicy().before_model(ctx, history)
    assert history == [{"role": "user", "content": "Câu hỏi gốc"}]
    assert (FINALIZE_SENTINEL in outbound[-1]["content"]) is spent
    assert outbound[:1] == history


def test_budget_refuses_a_tool_without_spending_submit_reserve():
    ctx = context(calls=7)

    def forbidden(*args):
        pytest.fail("spent budget must not dispatch a tool")

    result = BudgetPolicy().wrap_tool_call(ctx, forbidden, "search", {})
    assert not result.ok
    assert ctx.tools.calls == 7


@pytest.mark.parametrize("marker", DEGRADED_MARKERS)
def test_retry_handles_every_degraded_marker_even_when_ok(marker):
    ctx = context()
    args = {"doc_id": "source"}
    results = [ToolResult(True, marker), ToolResult(True, "evidence")]

    def call(name, received):
        assert name == "fetch_doc" and received is args
        ctx.tools.calls += 1
        return results.pop(0)

    result = Retry().wrap_tool_call(ctx, call, "fetch_doc", args)
    assert result.content == "evidence"
    assert ctx.tools.calls == 2
    assert ctx.state["retry_count"] == 1


@pytest.mark.parametrize("budget,initial,expected", [(8, 6, 1), (8, 5, 2),
                                                    (None, 0, 3)])
def test_retry_is_bounded_and_returns_the_actual_failure(budget, initial, expected):
    ctx = context(budget=budget, calls=initial)
    failure = ToolResult(False, "", error="timeout")

    def call(name, args):
        ctx.tools.calls += 1
        return failure

    assert Retry().wrap_tool_call(ctx, call, "fetch_doc", {}) is failure
    assert ctx.tools.calls == initial + expected
    assert ctx.state["retry_attempts"] == expected


@pytest.mark.parametrize("closed", [True, False])
def test_injection_guard_removes_repeated_or_unclosed_blocks(closed):
    hostile = BLOCK_START + "\n" + INJECTION_CANARY + "\n"
    content = "safe-before\n" + hostile + BLOCK_END + "\nsafe-middle\n" + hostile
    if closed:
        content += BLOCK_END + "\nsafe-after"
    original = ToolResult(True, content, error="preserve error")
    result = InjectionGuard().wrap_tool_call(context(), lambda *_: original, "fetch_doc", {})
    assert result.ok and result.error == original.error
    assert result.content.count(PLACEHOLDER) == 2
    assert "safe-before" in result.content and "safe-middle" in result.content
    assert ("safe-after" in result.content) is closed
    assert INJECTION_CANARY not in result.content
    assert original.content == content


def test_final_injection_sweep_only_changes_answer():
    claims = [{"text": INJECTION_CANARY, "doc_id": "source"}]
    report = {"answer": INJECTION_CANARY + " safe " + INJECTION_CANARY, "claims": claims}
    InjectionGuard().after_agent(context(), report)
    assert report["answer"] == " safe "
    assert report["claims"] is claims
    assert claims[0]["text"] == INJECTION_CANARY


def test_citation_checker_reattributes_to_observed_source_without_editing_text():
    unseen = doc("unseen", "Same quotation.\nUnseen additional evidence.")
    seen = doc("seen", "Same quotation.\nObserved additional evidence.")
    ctx = context([unseen, seen], [seen.body])
    claim = {"text": "Same quotation.", "doc_id": unseen.doc_id}
    report = {"claims": [claim]}
    CitationChecker().after_agent(ctx, report)
    assert claim == {"text": "Same quotation.", "doc_id": seen.doc_id}
    assert report["citations"] == [seen.doc_id]


def test_citation_checker_does_not_use_unseen_or_partially_observed_documents():
    source = doc("source", "A supported quotation.\nFull evidence.")
    ctx = context([source], ["A supported quotation."])
    claim = {"text": "A supported quotation.", "doc_id": "missing"}
    CitationChecker().after_agent(ctx, {"claims": [claim]})
    assert claim["doc_id"] == "missing"


def test_citation_checker_rejects_a_quote_spanning_two_lines():
    source = doc("source", "First line.\nSecond line.")
    claim = {"text": source.body, "doc_id": "missing"}
    CitationChecker().after_agent(context([source], [source.body]), {"claims": [claim]})
    assert claim["doc_id"] == "missing"


@pytest.mark.parametrize("claims", [None, {}, [], [None], [{"text": "invented"}],
                                    [{"text": ""}], [{"text": 12}]])
def test_critic_abstains_when_no_supported_claim_remains(claims):
    report = {"claims": claims, "answer": "unfounded", "citations": ["missing"], "abstain": False}
    Critic().after_agent(context(), report)
    assert report["claims"] == [] and report["citations"] == []
    assert report["abstain"] is True
    assert report["answer"] != "unfounded"


def test_critic_preserves_supported_words_and_does_not_judge_the_citation():
    ctx = context(observations=["Exact quotation, unchanged!"])
    claim = {"text": "Exact quotation, unchanged!", "doc_id": "wrong"}
    report = {"claims": [claim], "answer": "answer", "abstain": False}
    Critic().after_agent(ctx, report)
    assert report["claims"] == [claim]
    assert report["claims"][0] is claim
    assert report["abstain"] is False


def test_critic_splits_at_the_supported_join_not_an_earlier_conjunction():
    first = doc("first", "A và B are allowed.")
    second = doc("second", "C is required.")
    fused = first.body + " và " + second.body
    report = {"claims": [{"text": fused, "doc_id": "wrong"}], "abstain": False}
    ctx = context([first, second], [first.body, second.body])
    Critic().after_agent(ctx, report)
    assert report["claims"] == [
        {"text": first.body, "doc_id": first.doc_id},
        {"text": second.body, "doc_id": second.doc_id},
    ]
    assert all(c["text"] in fused for c in report["claims"])
    assert report["abstain"] is True


def test_critic_does_not_split_using_two_parts_of_the_same_document():
    source = doc("source", "First statement.\nSecond statement.")
    ctx = context([source], [source.body])
    report = {"claims": [{"text": "First statement. và Second statement.", "doc_id": "source"}]}
    Critic().after_agent(ctx, report)
    assert report["claims"] == []
    assert report["abstain"] is True


def test_critic_can_split_adjacent_conjunctions_without_rewriting_words():
    first = doc("first", "A is allowed và B needs approval.")
    second = doc("second", "C is required.")
    left, right = "A is allowed và", second.body
    report = {"claims": [{"text": left + " và " + right, "doc_id": "wrong"}]}
    Critic().after_agent(context([first, second], [first.body, second.body]), report)
    assert report["claims"] == [
        {"text": left, "doc_id": first.doc_id},
        {"text": right, "doc_id": second.doc_id},
    ]
    assert report["abstain"] is True


def test_critic_discards_a_claim_spanning_document_lines():
    source = doc("source", "First line.\nSecond line.")
    report = {"claims": [{"text": source.body, "doc_id": source.doc_id}]}
    Critic().after_agent(context([source], [source.body]), report)
    assert report["claims"] == []
    assert report["abstain"] is True
