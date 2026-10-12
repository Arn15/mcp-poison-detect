"""Registry context: which already-registered tools does a candidate description refer to?

Implicit poisoning must name a legitimate target tool and impose a precondition on it, so the detector
reads the candidate together with the tools it references. References are found by
  1) name matching (exact, case-insensitive, with snake_case / camelCase / spaced variants), then
  2) optional semantic matching (E5 embeddings) when no name matches, to catch paraphrased references.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence


def name_variants(name: str) -> set[str]:
    """'getEmailDetail' -> {'getemaildetail', 'get_email_detail', 'get-email-detail'}.

    The space-separated form ('get email detail') is deliberately excluded: it matches ordinary prose
    ("Search emails within ...") and creates false references."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).replace("-", "_").lower()
    words = [w for w in re.split(r"[_\s]+", snake) if w]
    out = {name.lower(), snake, "-".join(words)}
    if len(words) > 1:
        out.add("".join(words))
    return {v for v in out if len(v) >= 3}


def find_name_references(text: str, registry_names: Sequence[str], exclude: str | None = None) -> list[str]:
    """Registered tool names mentioned in `text`, in order of first mention."""
    low = text.lower()
    hits = []
    for name in registry_names:
        if exclude is not None and name == exclude:
            continue
        pos = None
        for v in name_variants(name):
            m = re.search(r"(?<![a-z0-9_])" + re.escape(v) + r"(?![a-z0-9_])", low)
            if m and (pos is None or m.start() < pos):
                pos = m.start()
        if pos is not None:
            hits.append((pos, name))
    return [n for _, n in sorted(hits)]


def find_references(text: str, registry: dict[str, str], exclude: str | None = None, top_k: int = 3,
                    semantic: Callable[[str, dict[str, str]], list[tuple[str, float]]] | None = None,
                    sim_threshold: float = 0.80) -> tuple[list[str], str]:
    """Return (referenced tool names, method) where method is 'name', 'semantic' or 'none'."""
    names = find_name_references(text, list(registry), exclude)
    if names:
        return names[:top_k], "name"
    if semantic is not None:
        cands = {k: v for k, v in registry.items() if k != exclude}
        scored = [(n, s) for n, s in semantic(text, cands) if s >= sim_threshold]
        if scored:
            return [n for n, _ in scored[:top_k]], "semantic"
    return [], "none"


def make_e5_semantic(model_name: str = "intfloat/e5-base-v2", device: str | None = None):
    """Build a semantic matcher using E5 (requires sentence-transformers). Caches tool embeddings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    cache: dict[str, object] = {}

    def embed(texts, prefix):
        return model.encode([prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True)

    def semantic(text: str, registry: dict[str, str]):
        keys = sorted(registry)
        missing = [k for k in keys if (k, registry[k]) not in cache]
        if missing:
            embs = embed([f"{k}: {registry[k]}" for k in missing], "passage: ")
            for k, e in zip(missing, embs):
                cache[(k, registry[k])] = e
        q = embed([text], "query: ")[0]
        scores = [(k, float(q @ cache[(k, registry[k])])) for k in keys]
        return sorted(scores, key=lambda x: -x[1])

    return semantic


def format_detector_input(candidate_name: str, candidate_desc: str, context: list[dict],
                          max_ctx_chars: int = 1200) -> tuple[str, str]:
    """(text_a, text_b) pair for a cross-encoder: candidate vs. referenced registered tools."""
    a = f"NEW TOOL {candidate_name}: {candidate_desc}"
    if not context:
        return a, "REGISTERED: none referenced"
    parts, used = [], 0
    for c in context:
        s = f"REGISTERED {c['tool_name']}: {c['description'] or '(no description)'}"
        if used + len(s) > max_ctx_chars:
            s = s[: max(0, max_ctx_chars - used)]
        parts.append(s)
        used += len(s)
        if used >= max_ctx_chars:
            break
    return a, " | ".join(parts)
