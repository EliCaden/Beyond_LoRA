# utils/targets.py
from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Set, Union

_CANON_BASE: List[str] = ["q", "k", "v", "o", "ffn1", "ffn2"]
_CANON_WITH_HEAD: List[str] = _CANON_BASE + ["head"]

_CANON_SET_BASE: Set[str] = set(_CANON_BASE)
_CANON_SET_WITH_HEAD: Set[str] = set(_CANON_WITH_HEAD)

_QKVO_CHARS = set("qkvo")

_SYNONYMS = {
    "query": "q",
    "q_proj": "q",
    "qproj": "q",
    "key": "k",
    "k_proj": "k",
    "kproj": "k",
    "value": "v",
    "v_proj": "v",
    "vproj": "v",
    "o": "o",
    "out": "o",
    "out_proj": "o",
    "outproj": "o",
    "proj": "o",
    "wo": "o",
    "dense": "o",
    "ffn": "mlp",
    "mlp": "mlp",
    "fc1": "ffn1",
    "w1": "ffn1",
    "up": "ffn1",
    "intermediate": "ffn1",
    "fc2": "ffn2",
    "w2": "ffn2",
    "down": "ffn2",
    "output": "ffn2",
    "classifier": "head",
    "cls": "head",
    "lm_head": "head",
}

_GROUPS = {"qv", "qkv", "attn", "all_attn", "all-attn", "mlp", "ffn", "all"}


def _is_nullish_str(x: str) -> bool:
    s = x.strip().lower()
    return s in {"none", "null", ""}


def _expand_compact_token(t: str) -> List[str]:
    t = t.strip().lower()
    if not t:
        return []
    if all(ch in _QKVO_CHARS for ch in t):
        return list(t)
    for suf in ("dense", "proj", "out", "outproj", "out_proj", "wo"):
        if t.startswith("qkv") and t.endswith(suf) and len(t) >= len("qkv") + len(suf):
            return ["q", "k", "v", suf]
        if t.startswith("qv") and t.endswith(suf) and len(t) >= len("qv") + len(suf):
            return ["q", "v", suf]
    return [t]


def _split_targets(x: str) -> List[str]:
    s = str(x).replace(",", " ").strip().lower()
    toks: List[str] = []
    for raw in s.split():
        toks.extend(_expand_compact_token(raw))
    return [t for t in toks if t]


def normalize_targets(
    target_modules: Union[Sequence[str], str, None],
    *,
    default: Sequence[str] = ("q", "v"),
    allow_head: bool = False,
) -> List[str]:
    canon_order = _CANON_WITH_HEAD if allow_head else _CANON_BASE
    canon_set = _CANON_SET_WITH_HEAD if allow_head else _CANON_SET_BASE

    if target_modules is None:
        toks: List[str] = list(default)
    elif isinstance(target_modules, str):
        if _is_nullish_str(target_modules):
            toks = list(default)
        else:
            toks = _split_targets(target_modules)
    else:
        toks = []
        for t in target_modules:
            toks.extend(_split_targets(str(t)))

    out: List[str] = []
    for t in toks:
        t = t.lower().strip()
        if not t:
            continue

        if t == "qv":
            out += ["q", "v"]
            continue
        if t == "qkv":
            out += ["q", "k", "v"]
            continue
        if t in {"attn", "all_attn", "all-attn"}:
            out += ["q", "k", "v", "o"]
            continue
        if t in {"mlp", "ffn"}:
            out += ["ffn1", "ffn2"]
            continue
        if t == "all":
            out += ["q", "k", "v", "o", "ffn1", "ffn2"]
            if allow_head:
                out += ["head"]
            continue

        t2 = _SYNONYMS.get(t, t)

        if t2 == "head" and not allow_head:
            raise ValueError("Target 'head' is not allowed here (allow_head=False). Use head_modules/explicit head config.")
        if t2 in canon_set:
            out.append(t2)
            continue

        raise ValueError(
            f"Unknown target module token {t!r}. "
            f"Allowed: {sorted(canon_set)} plus groups: {sorted(_GROUPS)} and common synonyms."
        )

    seen = set()
    final: List[str] = []
    for c in canon_order:
        if c in out and c not in seen:
            final.append(c)
            seen.add(c)
    return final


def normalize_target_modules(x: Any) -> Optional[str]:
    """
    Sweep-script behavior: canonicalize to a stable comma-separated subset of:
      q,k,v,o,ffn1,ffn2   (no 'head' here)

    Accepts group tokens and synonyms. If input normalizes to empty, returns the original input string
    (same as prior sweep_* behavior: "better than silently dropping").
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    sl = s.lower().strip()
    if sl in {"none", "null"}:
        return None

    raw = sl.replace(",", " ")
    toks: List[str] = []
    for part in raw.split():
        toks.extend(_expand_compact_token(part))

    expanded: List[str] = []
    for t in toks:
        t = t.strip().lower()
        if not t:
            continue
        if t == "qv":
            expanded += ["q", "v"]
            continue
        if t == "qkv":
            expanded += ["q", "k", "v"]
            continue
        if t in {"attn", "all_attn", "all-attn"}:
            expanded += ["q", "k", "v", "o"]
            continue
        if t in {"mlp", "ffn"}:
            expanded += ["ffn1", "ffn2"]
            continue
        if t == "all":
            expanded += ["q", "k", "v", "o", "ffn1", "ffn2"]
            continue

        expanded.append(_SYNONYMS.get(t, t))

    keep: List[str] = []
    seen = set()
    for c in _CANON_BASE:
        if c in expanded and c not in seen:
            keep.append(c)
            seen.add(c)

    return ",".join(keep) if keep else s


def target_tag(target_modules: Any) -> str:
    """
    Matches the sweep_* tag style: _tm<alnum>.
    """
    tm = normalize_target_modules(target_modules)
    if tm is None:
        return ""
    s = tm.strip().lower().replace(",", " ")
    s = "".join(s.split())
    s = re.sub(r"[^a-z0-9]+", "", s)
    return f"_tm{s}" if s else ""


def targets_key(targets: Sequence[str]) -> str:
    return "".join(list(targets))
