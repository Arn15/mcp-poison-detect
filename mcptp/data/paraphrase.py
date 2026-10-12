"""MCPTox-Para: meaning-preserving paraphrases of tool descriptions (no detector in the loop).

Purpose: MCPTox attacks are template-generated, so a bag-of-words model solves them (results/leakage_v1.json).
Paraphrasing both poisoned and benign/hard-negative tools keeps the semantics (who imposes what rule on which
tool) while removing template wording. Poisoned tools are paraphrased from `description_stripped`, so the
injection markers ("Ignore the previous instructions", <IMPORTANT>) are gone too.

Each output is checked: every referenced tool name kept verbatim, >= min_value_recall of quoted strings /
paths / numbers kept, length ratio in range, not a copy, no refusal / commentary.
"""
from __future__ import annotations

import random
import re

SYSTEM = ("You rewrite tool descriptions for MCP servers. The rewrites are used to train a security classifier, "
          "so they must be faithful: same meaning, same instructions, nothing added and nothing removed.")

USER_TMPL = """Rewrite the following tool description in a {style} style.

Rules:
- Keep the exact meaning and every instruction, including any instruction about calling, using, or changing other tools.
- Keep these tool names exactly as written: {names}
- Keep these values exactly as written: {values}
- Use different wording and sentence structure; do not copy phrases from the original.
- Do not add warnings, opinions, notes, or tags such as <IMPORTANT>.
- Output only the rewritten description as plain text.

Description:
{text}"""

REFUSAL = re.compile(r"\b(i can(?:no|')t|i'm sorry|i am sorry|as an ai|i won't|cannot assist|not able to help|"
                     r"malicious|harmful|i must decline|rewritten description:|here is|here's)\b", re.I)
_VALUE_RE = re.compile(r"'([^']{2,})'|\"([^\"]{2,})\"|((?:~|/)[\w.\-/~]+)|(\b[\w.+-]+@[\w-]+\.[\w.]+\b)|"
                       r"(https?://\S+)|(\b\d{2,}\b)")


def extract_values(text: str, exclude: set[str]) -> list[str]:
    vals = []
    for m in _VALUE_RE.finditer(text):
        v = next(g for g in m.groups() if g)
        v = v.strip().rstrip(".,;:")
        if v and v not in exclude and v not in vals:
            vals.append(v)
    return vals


def source_text(ex: dict) -> str:
    return ex["description_stripped"] if ex["label"] == 1 else ex["description"]


def tool_names_in(ex: dict) -> list[str]:
    """Referenced tool names that appear verbatim in the source text (these must survive paraphrasing)."""
    src = source_text(ex).lower()
    return [n for n in dict.fromkeys(ex.get("referenced_tools") or []) if n.lower() in src]


def build_prompt(ex: dict, style: str) -> tuple[str, list[str], list[str]]:
    text = source_text(ex)
    names = tool_names_in(ex)
    values = extract_values(text, set(names))
    user = USER_TMPL.format(style=style, text=text,
                            names=", ".join(f"'{n}'" for n in names) or "(none)",
                            values="; ".join(values[:12]) or "(none)")
    return user, names, values


def clean_output(out: str) -> str:
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    out = re.sub(r"^(?:rewritten description|description)\s*:\s*", "", out, flags=re.I).strip()
    out = out.strip().strip('"').strip()
    return re.sub(r"\s+", " ", out)


def check(src: str, out: str, names: list[str], values: list[str], min_value_recall: float,
          length_ratio=(0.5, 2.2)) -> dict:
    low = out.lower()
    names_ok = all(n.lower() in low for n in names)
    kept = [v for v in values if v.lower() in low]
    recall = len(kept) / len(values) if values else 1.0
    ratio = len(out) / max(1, len(src))
    copy = re.sub(r"\W", "", out.lower()) == re.sub(r"\W", "", src.lower())
    src_low = src.lower()
    refusal = any(m.group(0).lower() not in src_low for m in REFUSAL.finditer(out))
    valid = (names_ok and recall >= min_value_recall and length_ratio[0] <= ratio <= length_ratio[1]
             and not copy and not refusal and len(out) > 15)
    return {"valid": valid, "names_ok": names_ok, "value_recall": round(recall, 3), "len_ratio": round(ratio, 3),
            "copy": copy, "refusal": refusal}


def pick_style(styles: list[str], key: str, k: int, attempt: int) -> str:
    return random.Random(f"{key}:{k}:{attempt}").choice(styles)


# ---------------- generation backends ----------------
class EchoBackend:
    """CPU smoke test: swaps a few words, so the pipeline runs without a GPU."""
    name = "echo"

    def generate(self, prompts: list[str], seed: int = 0) -> list[str]:
        outs = []
        for p in prompts:
            text = p.split("Description:\n", 1)[1]
            outs.append(("In short: " + text.replace(" must ", " has to ").replace("Before", "Prior to")))
        return outs


class HFBackend:
    """Batched generation with transformers (fp16 on a single T4; Qwen3 non-thinking mode)."""

    def __init__(self, model_name: str, temperature: float, top_p: float, max_new_tokens: int, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(
            device if torch.cuda.is_available() else "cpu").eval()
        self.gen_kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                               pad_token_id=self.tok.pad_token_id)

    def _chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: list[str], seed: int = 0) -> list[str]:
        self.torch.manual_seed(seed)
        enc = self.tok([self._chat(p) for p in prompts], return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, **self.gen_kwargs)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def make_backend(pcfg: dict):
    if pcfg["backend"] == "echo":
        return EchoBackend()
    return HFBackend(pcfg["model"], pcfg["temperature"], pcfg["top_p"], pcfg["max_new_tokens"])
