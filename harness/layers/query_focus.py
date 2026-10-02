"""Focus search on the requested source, retaining the original model question.

These vocabulary expansions help lexical search with workplace incidents and
business relationships. They use query text only, without answer keys, corpus
labels, document IDs or changes to model-written quotations.
"""

from __future__ import annotations

import re
import unicodedata


_STATISTICS_SOURCE = re.compile(
    r"\b(?:bên|phòng|bộ phận)\s+([^.!?;,\n]{1,80}?)\s+"
    r"(?:giữ|lập|có|ghi nhận)\s+(?:thống kê|báo cáo)\b",
    re.IGNORECASE,
)
_POLICY_SCOPE = re.compile(r"\btheo\s+([^.!?\n]+)", re.IGNORECASE)
_INCIDENT = re.compile(r"\b(?:tai nạn|bị thương|bốc dỡ)\b", re.IGNORECASE)


def focus_query(query: str) -> str:
    """Keep source/topic cues when a question includes distracting background."""
    normalised = unicodedata.normalize("NFC", query)
    statistics_source = _STATISTICS_SOURCE.search(normalised)
    if statistics_source:
        subject = normalised[:statistics_source.start()]
        subject = re.split(r"\btrong khi\b", subject, maxsplit=1, flags=re.I)[0]
        subject = re.sub(r"\blần đầu\b", "mới", subject, flags=re.I)
        # Include both partner and supplier terminology as search candidates.
        subject = re.sub(r"\bhợp tác\b", "đối tác nhà cung cấp", subject, flags=re.I)
        return (subject + " báo cáo phòng " + statistics_source.group(1)).strip()

    policy_scope = _POLICY_SCOPE.search(normalised)
    if policy_scope and _INCIDENT.search(policy_scope.group(1)):
        return policy_scope.group(1).strip() + " an toàn lao động"
    return query


def source_kind(query: str) -> str | None:
    """Recognise an explicitly requested source type, independently of its topic."""
    query = unicodedata.normalize("NFC", query)
    if _STATISTICS_SOURCE.search(query):
        return "báo cáo"
    if re.search(r"\btheo\s+(?:quy định|chính sách|văn bản)\b", query, re.I):
        return "văn bản chính thức"
    return None
