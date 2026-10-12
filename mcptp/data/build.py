"""Assemble detector examples: candidate tool + server registry + referenced-tool context + label."""
from __future__ import annotations

from .context import find_references
from .hard_negatives import templated_hard_negatives
from .mcptox import load_mcptox, normalize_ws, stable_id, strip_injection_markers


def build_examples(cfg: dict, raw_dir, semantic=None) -> tuple[list[dict], dict]:
    dcfg, ccfg = cfg["data"], cfg["context"]
    tables = load_mcptox(raw_dir)
    benign = tables["benign"].drop_duplicates(["server", "tool_name"]).reset_index(drop=True)
    poison = tables["poison"]
    if not dcfg.get("include_explicit", True):
        poison = poison[poison.is_implicit]

    registries = {s: dict(zip(g.tool_name, g.description)) for s, g in benign.groupby("server")}
    top_k, thr = ccfg["top_k"], ccfg["sim_threshold"]
    sem = semantic if ccfg.get("semantic_match") else None

    def with_context(rec: dict, exclude: str | None) -> dict:
        reg = registries[rec["server"]]
        refs, method = find_references(rec["description"], reg, exclude=exclude, top_k=top_k,
                                       semantic=sem, sim_threshold=thr)
        rec["referenced_tools"] = refs
        rec["ref_method"] = method
        rec["context"] = [{"tool_name": r, "description": reg[r]} for r in refs]
        rec["registry_size"] = len(reg) - (1 if exclude in reg else 0)
        return rec

    examples = []
    for r in poison.itertuples():
        examples.append(with_context({
            "id": r.id, "server": r.server, "kind": "poison", "label": 1, "tool_name": r.tool_name,
            "description": r.description, "description_stripped": r.description_stripped,
            "paradigm": r.paradigm, "risk": r.risk, "is_implicit": bool(r.is_implicit),
            "has_marker": bool(r.has_marker), "base_key": r.base_key, "seed_asr": r.seed_asr,
            "queries": list(r.queries), "is_hard_negative": False, "hn_family": None}, exclude=None))

    for r in benign.itertuples():
        rec = with_context({
            "id": r.id, "server": r.server, "kind": "benign", "label": 0, "tool_name": r.tool_name,
            "description": r.description, "description_stripped": strip_injection_markers(r.description),
            "paradigm": None, "risk": None, "is_implicit": None, "has_marker": False,
            "base_key": r.id, "seed_asr": None, "queries": []}, exclude=r.tool_name)
        rec["is_hard_negative"] = rec["ref_method"] == "name"   # mined: real benign tool naming a sibling
        rec["hn_family"] = "mined" if rec["is_hard_negative"] else None
        examples.append(rec)

    n_hn = int(dcfg.get("hard_negatives_per_server", 8))
    for server in sorted(registries):
        for h in templated_hard_negatives(server, sorted(registries[server]), n_hn, cfg["project"]["seed"],
                                              sibling_descriptions=registries[server]):
            desc = normalize_ws(h["description"])
            examples.append(with_context({
                "id": "h_" + stable_id(server, h["tool_name"], desc), "server": server, "kind": "hard_negative",
                "label": 0, "tool_name": h["tool_name"], "description": desc,
                "description_stripped": strip_injection_markers(desc), "paradigm": None, "risk": None,
                "is_implicit": None, "has_marker": "<important>" in desc.lower(), "base_key": h["tool_name"],
                "seed_asr": None, "queries": [], "is_hard_negative": True, "hn_family": h["hn_family"]},
                exclude=None))

    ids = [e["id"] for e in examples]
    assert len(ids) == len(set(ids)), "duplicate example ids"
    return examples, tables
