"""Query router -> retrieval scope + structured keys. Primary path is an LLM
self-query (LangChain structured output) mapping the question to {intent,
document, note_numbers, tags, fields} by meaning (base/revised, note 1, a tag,
a field) — no filename hardcoding. Falls back to deterministic rules with no key
or on error. Only picks WHAT to retrieve; the answerer still grounds + cites."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

from src.ingest.typing_rules import TAG_RE

Intent = Literal["changed", "content"]

_BASE_HINTS = ("base", "original", "old", "previous", "rev a")
_REV_HINTS = ("revised", "revision", "updated", "rev b", "latest")
_STRONG_CHANGE = ("changed", "change", "added", "remove", "removed", "modified",
                  "modification", "difference", "differ", "delta", "diff")
_SUMMARY = ("comparison", "compare", "summary", "summarize", "overview", "detail", "what changed")
_BOTH = ("both", "each", "compare", " vs", "versus")
_FIELD_KW = {"duty": "DUTY", "service": "SERVICE", "flow rate": "FLOW RATE", "flow": "FLOW RATE",
             "tag number": "TAG NUMBER", "material": "MATERIAL", "vendor": "VENDOR",
             "quantity": "QUANTITY", "vessel trim": "VESSEL TRIM"}
_KNOWN_FIELDS = sorted(set(_FIELD_KW.values()))
# "list ALL / every / summarize" -> enumerate the full delta set, not just top-k
_AGG_RE = re.compile(r"\b(all|every|each|entire|complete|everything|list|summar\w*)\b", re.I)
_CT_KW = [("added", r"\badd"), ("removed", r"\b(remov|delet)"),
          ("modified", r"\bmodif"), ("moved", r"\bmove")]
_CHANGE_TYPES = {"added", "removed", "modified", "moved"}


@dataclass
class RouteDecision:
    intent: Intent
    pid_filter: list[str]
    boost_delta: bool
    keys: list[str] = field(default_factory=list)
    aggregate: bool = False                          # enumerate the full matching delta set
    change_types: list[str] = field(default_factory=list)   # restrict aggregate to these (empty = all)


def _aggregate(query: str) -> tuple[bool, list[str]]:
    return bool(_AGG_RE.search(query)), [t for t, pat in _CT_KW if re.search(pat, query, re.I)]


def _regex_keys(query: str) -> list[str]:
    q = query.lower()
    keys = [f"note:{int(m.group(1))}" for m in re.finditer(r"\bnote\s+(\d{1,2})\b", q)]
    keys += [f"tag:{t}" for t in TAG_RE.findall(query)]
    keys += [f"field:{lab}" for kw, lab in _FIELD_KW.items() if kw in q]
    return list(dict.fromkeys(keys))


# --- deterministic fallback (also used when no API key) ---
def _rule_route(query, pid_a, pid_b, family_a, family_b) -> RouteDecision:
    q = query.lower()
    keys = _regex_keys(query)
    strong = any(w in q for w in _STRONG_CHANGE)
    summary = any(w in q for w in _SUMMARY)
    intent: Intent = "changed" if (strong or (summary and not keys)) else "content"
    if intent == "changed":
        agg, ctypes = _aggregate(query)
        return RouteDecision("changed", [pid_a, pid_b], True, keys, aggregate=agg, change_types=ctypes)

    def fam(f):
        return [t for t in re.split(r"[-_ ]+", (f or "").lower()) if len(t) > 2
                and t not in {"gas", "compressor", "the", "pid", "id", "hp", "stage"}]
    both = any(w in q for w in _BOTH)
    if not both and any(t in q for t in fam(family_b)):
        pid_filter = [pid_b]
    elif not both and any(t in q for t in fam(family_a)):
        pid_filter = [pid_a]
    elif not both and any(h in q for h in _REV_HINTS):
        pid_filter = [pid_b]
    elif not both and any(h in q for h in _BASE_HINTS):
        pid_filter = [pid_a]
    else:
        pid_filter = [pid_a, pid_b]
    return RouteDecision("content", pid_filter, False, keys)


# --- LLM self-query router (primary) ---
def _llm_route(query, pid_a, pid_b, family_a, family_b) -> RouteDecision:
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from src.chat.llm import build_chat_model
    from src.observability.tracing import langchain_callbacks

    class Route(BaseModel):
        intent: Literal["changed", "content"] = Field(
            description="'changed' if the user asks what changed/added/removed/modified/differs "
                        "between the two revisions; otherwise 'content'.")
        document: Literal["base", "revised", "both"] = Field(
            description="Which document the question targets. Map the user's wording to base vs "
                        "revised by MEANING (subject/name/old-new), not by filename.")
        note_numbers: list[int] = Field(default_factory=list,
                                        description="Explicit note numbers referenced, e.g. 'note 1' -> [1].")
        tags: list[str] = Field(default_factory=list,
                                description="Instrument/equipment tags referenced verbatim, e.g. 26-PIT-9055.")
        fields: list[str] = Field(default_factory=list,
                                  description=f"Title-block fields referenced, from: {_KNOWN_FIELDS}.")
        aggregate: bool = Field(default=False,
                                description="true if the user wants ALL/every matching item enumerated "
                                            "or a full/complete summary, not just the most relevant few.")
        change_types: list[str] = Field(default_factory=list,
                                        description="specific change types requested: added, removed, "
                                                    "modified, moved. Empty = all types.")

    sys = (f"You route a question about two revisions of an engineering document.\n"
           f"BASE (original) document subject: {family_a}\n"
           f"REVISED (new) document subject: {family_b}\n"
           f"Return the routing decision. If the user names a document by its subject/name "
           f"(e.g. a word from its subject) or by old/new, resolve it to base or revised.")
    model = build_chat_model().with_structured_output(Route)
    r: Route = model.invoke([SystemMessage(sys), HumanMessage(query)],
                            config={"callbacks": langchain_callbacks()})

    pid_filter = {"base": [pid_a], "revised": [pid_b], "both": [pid_a, pid_b]}[r.document]
    if r.intent == "changed":
        pid_filter = [pid_a, pid_b]
    keys = [f"note:{n}" for n in r.note_numbers]
    keys += [f"tag:{t}" for t in r.tags]
    keys += [f"field:{f.upper() if f.upper() in _KNOWN_FIELDS else _FIELD_KW.get(f.lower(), f.upper())}"
             for f in r.fields]
    keys = list(dict.fromkeys(keys + _regex_keys(query)))   # union w/ regex (belt + suspenders)
    agg, ctypes = _aggregate(query)
    aggregate = r.aggregate or agg
    change_types = sorted(set(ctypes) | {c for c in r.change_types if c in _CHANGE_TYPES})
    return RouteDecision(r.intent, pid_filter, r.intent == "changed", keys,
                         aggregate=aggregate, change_types=change_types)


def route(query: str, pid_a: str, pid_b: str, cfg,
          family_a: str = "", family_b: str = "") -> RouteDecision:
    use_llm = getattr(cfg.chat, "router", {})
    use_llm = (use_llm.get("use_llm", True) if hasattr(use_llm, "get") else True)
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            return _llm_route(query, pid_a, pid_b, family_a, family_b)
        except Exception:
            pass   # fall back to rules (observability captures the error upstream)
    return _rule_route(query, pid_a, pid_b, family_a, family_b)
