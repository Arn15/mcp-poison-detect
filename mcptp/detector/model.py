"""Detector inputs and model.

Plain (baseline B3):  "[CLS] NEW TOOL <name>: <description> [SEP]"
Context-aware (ours): "[CLS] NEW TOOL <name>: <description> [SEP] REGISTERED <t1>: <desc1> | ... [SEP]"
The context side holds the already-registered tools the candidate references (mcptp.data.context), so the
cross-encoder can check whether the candidate imposes rules on a tool it does not own.
"""
from __future__ import annotations

import random

from ..data.context import format_detector_input


def detector_text(ex: dict, use_context: bool, field: str = "description", context_override=None,
                  max_ctx_chars: int = 1200):
    ctx = (ex.get("context") or []) if context_override is None else context_override
    a, b = format_detector_input(ex["tool_name"], ex[field], ctx, max_ctx_chars)
    return (a, b) if use_context else (a, None)


def shuffled_contexts(examples: list[dict], seed: int = 0) -> list[list[dict]]:
    """Control: give each example the context of a random example from a *different* server."""
    rng = random.Random(seed)
    pool = [e for e in examples if e.get("context")]
    out = []
    for e in examples:
        cands = [p for p in pool if p["server"] != e["server"]] or pool
        out.append(rng.choice(cands)["context"] if cands else [])
    return out


def build_model(backbone: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(backbone)
    # transformers 5 keeps the checkpoint dtype by default (deberta-v3 ships fp16). Master weights must be fp32
    # for AMP: fp16 is applied only inside autocast, and GradScaler refuses fp16 gradients.
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2, dtype=torch.float32)
    return tok, model.float()
