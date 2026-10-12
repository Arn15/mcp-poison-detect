"""Load MCPTox (Wang et al., AAAI 2026) into flat tables.

Raw layout (commit f85189f of github.com/zhiqiangwang4/MCPTox-Benchmark):
  response_all.json["servers"][name] = {
      tool_names, clean_system_promot (benign tool list as text), clean_querys, server_url,
      malicious_instance: [{poisoned_tool, metadata{paradigm, security risk}, datas:[{query, response, label{model: outcome}}]}]}
  pure_tool.json = 485 base poisoned tools (instances are these plus injection-phrase variants).

Paradigms (MCPTox paper): Template-1 explicit trigger / function hijacking; Template-2 implicit trigger /
function hijacking; Template-3 implicit trigger / parameter tampering. Implicit = Template-2 and Template-3.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

MCPTOX_URL = "https://github.com/zhiqiangwang4/MCPTox-Benchmark.git"
MCPTOX_COMMIT = "f85189f9ad12504c197c7f920ab818a40657b1fa"
IMPLICIT_PARADIGMS = {"Template-2", "Template-3"}

_TOOL_RE = re.compile(r"Tool:[ \t]*(?P<name>[^\n]+?)[ \t]*\nDescription:(?P<desc>.*?)\nArguments:(?P<args>.*?)(?=\n\s*\nTool:|\Z)", re.S)

# Injection phrases MCPTox adds on top of base attacks. Kept so we can build a "stripped" view and test
# whether a detector relies on these surface cues (MCP-Poison-Bench keyword-circularity concern).
INJECTION_MARKERS = [
    r"Ignore the previous instructions\s*and do the following:\s*",
    r"<IMPORTANT>This description's priority is higher than the user's query[^<]*?(?:</IMPORTANT>|(?=\.\s)|$)",
    r"This description's priority is higher than the user's query(?:/request)?\.?\s*",
    r"</?IMPORTANT>",
    r"This is NOT optional!?\s*",
]
_MARKER_RE = re.compile("|".join(f"(?:{m})" for m in INJECTION_MARKERS), re.I)


def strip_injection_markers(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _MARKER_RE.sub(" ", text)).strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def stable_id(*parts: str, n: int = 12) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:n]


def ensure_mcptox(raw_dir: str | Path) -> Path:
    """Return the MCPTox directory, cloning it at the pinned commit if missing."""
    d = Path(raw_dir) / "mcptox"
    if (d / "response_all.json").exists():
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", MCPTOX_URL, str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", MCPTOX_COMMIT], check=True)
    return d


def parse_tool_block(text: str) -> list[dict]:
    """Parse 'Tool: x\\nDescription: ...\\nArguments: ...' blocks."""
    out = []
    for m in _TOOL_RE.finditer(text):
        out.append({"tool_name": m["name"].strip(), "description": normalize_ws(m["desc"]),
                    "arguments": normalize_ws(m["args"])})
    return out


def load_mcptox(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Return {'servers', 'benign', 'poison'} DataFrames."""
    d = ensure_mcptox(raw_dir)
    with open(d / "response_all.json") as f:
        raw = json.load(f)

    servers, benign, poison = [], [], []
    for sname in sorted(raw["servers"]):
        sv = raw["servers"][sname]
        tools = parse_tool_block(sv["clean_system_promot"])
        names = {t["tool_name"] for t in tools}
        missing = set(sv["tool_names"]) - names
        if missing:
            raise ValueError(f"{sname}: could not parse descriptions for {sorted(missing)}")
        servers.append({"server": sname, "server_url": sv.get("server_url", ""), "n_tools": len(tools),
                        "tool_names": sorted(names), "queries": sv.get("clean_querys", [])})
        for t in tools:
            benign.append({"id": "b_" + stable_id(sname, t["tool_name"]), "server": sname, **t})

        for mi in sv["malicious_instance"]:
            parsed = parse_tool_block(mi["poisoned_tool"] + "\n")
            if not parsed:
                continue
            t = parsed[0]
            meta = mi.get("metadata", {})
            labels = {}
            queries = []
            for dat in mi["datas"]:
                queries.append(dat.get("query", ""))
                for model, lab in (dat.get("label") or {}).items():
                    labels.setdefault(model, []).append(lab)
            flat = [lab for v in labels.values() for lab in v if lab != "None"]
            asr = sum(lab == "Success" for lab in flat) / len(flat) if flat else None
            stripped = strip_injection_markers(t["description"])
            poison.append({
                "id": "p_" + stable_id(sname, t["tool_name"], t["description"]),
                "server": sname, "tool_name": t["tool_name"], "description": t["description"],
                "description_stripped": stripped, "arguments": t["arguments"],
                "paradigm": meta.get("paradigm"), "risk": meta.get("security risk"),
                "is_implicit": meta.get("paradigm") in IMPLICIT_PARADIGMS,
                "has_marker": stripped != normalize_ws(t["description"]),
                # base attack = same text once injection markers are removed
                "base_key": stable_id(sname, t["tool_name"], stripped.lower()),
                "queries": queries, "seed_asr": asr, "n_judged": len(flat),
                "labels": labels,
            })

    poison_df = pd.DataFrame(poison).drop_duplicates("id").reset_index(drop=True)
    return {"servers": pd.DataFrame(servers), "benign": pd.DataFrame(benign), "poison": poison_df}
