"""Check retrieval improvements with new documents and unchanged tool contracts."""

from types import SimpleNamespace
import json
import unicodedata

import pytest

from arena.corpus import Corpus, Doc
from arena.tools import ToolResult
from harness.agent import AgentContext
from harness.layers.citation_checker import CitationChecker


def search(query, docs, k=1):
    corpus = Corpus(docs)
    brief = {"question_vi": query}
    ctx = AgentContext(brief=brief, tools=SimpleNamespace(calls=0), trace=None, corpus=corpus)
    arguments = {"query": query, "k": k}
    captured = []

    def call(name, args):
        assert name == "search"
        captured.append(args)
        ctx.tools.calls += 1
        return ToolResult(True, json.dumps([
            {"doc_id": d.doc_id, "title": d.title, "snippet": d.body}
            for d in corpus.search(args["query"], args["k"])
        ], ensure_ascii=False))

    result = CitationChecker().wrap_tool_call(ctx, call, "search", arguments)
    assert ctx.brief == brief == {"question_vi": query}
    assert arguments == {"query": query, "k": k}
    assert captured[0]["k"] == k
    assert ctx.tools.calls == 1
    return " ".join(hit["doc_id"] for hit in json.loads(result.content)), captured[0]["query"]


@pytest.mark.parametrize("unicode_form", ["NFC", "NFD"])
def test_policy_scope_retrieves_safety_source_over_support_ticket(unicode_form):
    query = unicodedata.normalize(unicode_form,
        "Một ticket hỗ trợ ghi nhận tai nạn. Theo quy định phòng chống tai nạn, phải báo ai?")
    docs = [
        Doc("ticket", "Ticket hỗ trợ", "Ticket hỗ trợ ghi nhận tai nạn, cần hỗ trợ.", ()),
        Doc("policy", "An toàn lao động", "An toàn lao động: phải báo cho quản đốc trong 6 giờ.", ()),
    ]
    result, _ = search(query, docs)
    assert result == "policy"


def test_statistics_source_retains_subject_and_discards_contrasting_ticket():
    query = ("Một đơn vị ký hợp tác lần đầu hỏi về hồ sơ, trong khi ticket khách hàng bị trễ. "
             "Phòng Mua sắm giữ thống kê. Hãy nêu số trường hợp.")
    docs = [
        Doc("ticket", "Ticket khách hàng bị trễ", "Ticket khách hàng bị trễ liên quan hồ sơ.", ()),
        Doc("report", "Nhà cung cấp mới — Báo cáo", "Phòng Mua sắm ghi nhận 9 trường hợp.", ()),
    ]
    result, outbound = search(query, docs)
    assert result == "report"
    assert "Mua sắm" in outbound
    assert "ticket" not in outbound


@pytest.mark.parametrize("name,args", [
    ("search", {"query": "Điều kiện hoàn tiền cho đơn giao trễ", "k": 3}),
    ("search", {"query": 123, "k": 3}),
    ("fetch_doc", {"doc_id": "a-source"}),
    ("calc", {"expression": "2+2"}),
])
def test_unrelated_calls_keep_their_arguments_and_result(name, args):
    result = ToolResult(True, "unaltered evidence")
    captured = []

    def call(received_name, received_args):
        captured.append((received_name, received_args))
        return result

    assert CitationChecker().wrap_tool_call(None, call, name, args) is result
    assert captured == [(name, args)]
    assert captured[0][1] is args


def test_statistics_source_does_not_cross_sentences_or_change_unrelated_questions():
    query = "Phòng Vận hành đang xử lý ticket. Một nhóm khác có thống kê chưa?"
    result, outbound = search(query, [Doc("ticket", "Ticket vận hành", query, ())])
    assert result == "ticket"
    assert outbound == query


@pytest.mark.parametrize("question,titles", [
    ("Theo quy định kiểm kê, phải báo ai?", ["Kiểm kê — Hỏi & Đáp", "Kiểm kê — Văn bản chính thức"]),
    ("Phòng Mua sắm giữ thống kê. Hãy nêu số vụ.", ["Kiểm kê — Văn bản chính thức", "Kiểm kê — Báo cáo"]),
])
def test_explicit_source_type_prioritises_existing_hits_without_changing_evidence(question, titles):
    hits = [{"doc_id": f"source-{i}", "title": title, "snippet": f"Original evidence {i}."}
            for i, title in enumerate(titles)]
    original = ToolResult(True, json.dumps(hits, ensure_ascii=False), error="preserve metadata")
    result = CitationChecker().wrap_tool_call(None, lambda *_: original, "search", {"query": question})
    assert json.loads(result.content) == hits[::-1]
    assert result.ok and result.error == original.error
    assert json.loads(original.content) == hits


@pytest.mark.parametrize("failure", [
    ToolResult(False, "", error="timeout"),
    ToolResult(True, "[TRUNCATED: incomplete JSON"),
    ToolResult(True, "[NOISE: corrupted search"),
])
def test_source_priority_preserves_failed_or_degraded_results_for_retry(failure):
    args = {"query": "Theo quy định kiểm kê, phải báo ai?"}
    assert CitationChecker().wrap_tool_call(None, lambda *_: failure, "search", args) is failure
