"""Measure attack success rate (ASR) of a candidate description on a target agent. Defensive measurement."""
from __future__ import annotations

from collections import Counter

from .agent import build_agent_prompt, classify, extract_tool_calls


def measure_asr(agent_backend, registry_block: str, candidate_block: str, queries: list[str], spec: dict,
                seed: int = 0) -> dict:
    """Run the agent on each query with the candidate tool present; return outcome rates."""
    if not queries:
        return {"asr": 0.0, "n": 0, "outcomes": {}}
    system = build_agent_prompt(registry_block, candidate_block)
    reswhich = agent_backend.generate(list(queries), seed=seed, system=system, max_new_tokens=160)
    outc = Counter()
    for q, resp in zip(queries, reswhich):
        outc[classify(extract_tool_calls(resp), spec, q)] += 1
    n = sum(outc.values())
    return {"asr": outc["success"] / n if n else 0.0, "n": n, "outcomes": dict(outc)}


def candidate_block(tool_name: str, description: str, arguments: str = "- No arguments") -> str:
    """Format a candidate tool as it appears in the agent's tool list (name, description, arguments)."""
    return f"Tool: {tool_name}\nDescription: {description}\nArguments:\n{arguments}\n"
