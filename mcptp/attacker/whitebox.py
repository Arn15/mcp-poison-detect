"""White-box word-substitution attack on a fine-tuned detector (HotFlip / GCG style). Defensive evaluation.

Purpose: stress-test the claim that fine-tuned classifiers "hold" against adaptive attackers. The black-box
LLM attacker (optimize.py) found zero genuine evasions, which could mean the classifier is robust or that a
4B/14B attacker is too weak. Here the attacker sees the detector's gradients, which is the strongest
practical threat model for a text classifier.

Starting point: control-verified *working* attacks (data/attacks/verified.jsonl), so every input already
hijacks the agent. Each step:
  1. gradient of the detector's logit margin (logit_poisoned - logit_benign) w.r.t. a one-hot relaxation of
     every editable word token -> top-k replacement words per position (first-order estimate, HotFlip)
  2. sample B single-word swaps from those top-k lists, score them EXACTLY on the re-tokenized text (GCG)
  3. keep the best swap if it lowers the margin
until the margin falls below the deployable threshold (val negatives at the target FPR), the word budget is
spent, or no swap helps for `patience` steps.

Constraints keep the attack meaningful and the payload intact:
  * whole-word swaps only: a word that is a single word-initial token is replaced by another single-token,
    alphabetic, word-initial vocabulary item (so the text stays words, never sub-word fragments)
  * protected strings are never touched: the candidate's own name, the target (legit) tool, the malicious
    tool names and the injected values from the attack spec
  * at most max(min_budget, max_frac * n_words) distinct words differ from the starting description
Evading rewrites must still pass the agent no-poison control (scripts/attack_whitebox.py, verify phase)
before they count as genuine evasions.

Only the plain (no-context) detector input is supported: "NEW TOOL <name>: <description>".
"""
from __future__ import annotations

import re

import numpy as np

SP_PREFIX = "▁"   # sentencepiece word-start marker (DeBERTa-v3, XLM-R / multilingual-E5)


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def protected_spans(text: str, strings) -> list[tuple[int, int]]:
    """Character spans of every case-insensitive occurrence of each protected string in text."""
    spans = []
    low = text.lower()
    for s in strings or []:
        s = (s or "").strip()
        if len(s) < 2:
            continue
        for m in re.finditer(re.escape(s.lower()), low):
            spans.append((m.start(), m.end()))
    return spans


def overlaps(span: tuple[int, int], spans) -> bool:
    s, e = span
    return any(s < pe and ps < e for ps, pe in spans)


def piece_style(pieces) -> str:
    """'sp' (sentencepiece, word-initial pieces start with U+2581) or 'wp' (wordpiece, continuations start ##)."""
    return "sp" if any(p.startswith(SP_PREFIX) for p in pieces[:5000] if p) else "wp"


def piece_word(piece: str, style: str) -> str | None:
    """The word a vocabulary piece stands for if it is word-initial, else None."""
    if style == "sp":
        return piece[1:] if piece.startswith(SP_PREFIX) else None
    return None if piece.startswith("##") else piece


def is_glue(piece: str | None, style: str) -> bool:
    """True if this piece continues the previous word (alphanumeric continuation)."""
    if piece is None:
        return False
    if style == "sp":
        return not piece.startswith(SP_PREFIX) and piece[:1].isalnum()
    return piece.startswith("##")


def allowed_vocab(pieces, style: str, special_ids=(), min_len: int = 2) -> np.ndarray:
    """Replacement vocabulary: word-initial, ASCII-alphabetic, at least min_len letters, not special."""
    ok = np.zeros(len(pieces), dtype=bool)
    for i, p in enumerate(pieces):
        w = piece_word(p or "", style)
        ok[i] = bool(w) and w.isascii() and w.isalpha() and len(w) >= min_len
    for i in special_ids:
        if 0 <= i < len(ok):
            ok[i] = False
    return ok


def editable_positions(pieces, offsets, text: str, desc_start: int, prot, style: str) -> list[tuple[int, int, int]]:
    """Token positions that are whole single-token alphabetic words inside the description and not protected.

    Returns (token_index, char_start, char_end) with char_start past any leading whitespace.
    """
    out = []
    n = len(pieces)
    for i, (p, (s, e)) in enumerate(zip(pieces, offsets)):
        if e <= s:
            continue      # special token
        while s < e and text[s].isspace():
            s += 1        # some tokenizers include the preceding space in a word-initial piece's offset
        if s < desc_start:
            continue
        w = piece_word(p or "", style)
        if not w or not w.isalpha():
            continue
        if i + 1 < n and is_glue(pieces[i + 1], style):
            continue      # multi-token word: swapping one piece would create a fragment
        if text[s:e].lower() != w.lower():
            continue      # offsets disagree with the piece (normalization); skip rather than guess
        if overlaps((s, e), prot):
            continue
        out.append((i, s, e))
    return out


def match_case(new: str, old: str) -> str:
    if old.isupper() and len(old) > 1:
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    return new.lower()


def replace_span(text: str, s: int, e: int, word: str) -> str:
    return text[:s] + match_case(word, text[s:e]) + text[e:]


def word_index(desc: str, char_pos: int) -> int:
    """Index of the whitespace-delimited word starting at char_pos in desc."""
    return len(desc[:char_pos].split())


def changed_words(orig: str, cur: str) -> set[int]:
    a, b = orig.split(), cur.split()
    if len(a) != len(b):   # whole-word swaps keep the word count; fall back to a conservative count
        return set(range(max(len(a), len(b))))
    return {i for i, (x, y) in enumerate(zip(a, b)) if x != y}


def word_budget(n_words: int, max_frac: float, min_budget: int) -> int:
    return max(int(min_budget), int(np.floor(max_frac * n_words)))


# ----------------------------------------------------------------------------- the attack
class WhiteBoxAttacker:
    """Gradient-guided word substitution against a sequence classifier with a 2-way head.

    margin_fn(list[str] descriptions) -> np.ndarray of exact logit margins for those descriptions (e.g.
    LoadedDetector.margin). The gradient pass uses the same model and the same "NEW TOOL <name>: " prefix.
    """

    def __init__(self, model, tok, device, max_len: int = 384, min_word_len: int = 2):
        import torch

        self.model, self.tok, self.device, self.max_len = model, tok, device, max_len
        self.model.eval()
        self.model.requires_grad_(False)   # only the one-hot relaxation needs a gradient
        self.pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.style = piece_style(self.pieces)
        allowed = allowed_vocab(self.pieces, self.style, tok.all_special_ids, min_word_len)
        self.E = model.get_input_embeddings().weight          # (V', H); V' can exceed len(tok)
        mask = np.zeros(self.E.shape[0], dtype=bool)
        n = min(len(allowed), len(mask))
        mask[:n] = allowed[:n]
        allowed = mask
        self.allowed = torch.tensor(mask, device=self.E.device)
        self.words = {i: piece_word(self.pieces[i], self.style) for i in np.flatnonzero(allowed)}

    @staticmethod
    def detector_text(name: str, desc: str) -> tuple[str, int]:
        a = f"NEW TOOL {name}: {desc}"    # == mcptp.data.context.format_detector_input(...)[0]
        return a, len(a) - len(desc)

    def grad_topk(self, name: str, desc: str, prot_strings, k: int):
        """Return (positions [(tok_i, desc_s, desc_e)], topk ids LongTensor (P, k)) for editable words."""
        import torch

        text, d0 = self.detector_text(name, desc)
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_offsets_mapping=True,
                       return_tensors="pt")
        ids = enc["input_ids"][0]
        offs = enc.pop("offset_mapping")[0].tolist()
        pieces = self.tok.convert_ids_to_tokens(ids.tolist())
        prot = protected_spans(text, prot_strings)
        pos = editable_positions(pieces, offs, text, d0, prot, self.style)
        if not pos:
            return [], None
        idx = torch.tensor([p[0] for p in pos])
        ids_d = ids.to(self.E.device)
        onehot = torch.zeros(len(pos), self.E.shape[0], device=self.E.device, dtype=self.E.dtype)
        onehot[torch.arange(len(pos)), ids_d[idx]] = 1.0
        onehot.requires_grad_(True)
        emb = self.E[ids_d].detach().clone()
        emb[idx.to(self.E.device)] = onehot @ self.E
        kw = {k_: v.to(self.device) for k_, v in enc.items() if k_ != "input_ids"}
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs_embeds=emb[None].to(self.device), **kw).logits.float()
        (logits[0, 1] - logits[0, 0]).backward()
        score = -onehot.grad.detach()                        # direction that lowers the margin
        score[:, ~self.allowed] = -float("inf")
        score[torch.arange(len(pos)), ids_d[idx]] = -float("inf")
        top = score.topk(min(k, int(self.allowed.sum())), dim=1).indices.cpu()
        return [(i, s - d0, e - d0) for i, s, e in pos], top

    def run(self, name: str, desc: str, prot_strings, margin_fn, threshold: float, steps: int = 60,
            batch: int = 64, topk: int = 32, max_frac: float = 0.15, min_budget: int = 2, patience: int = 15,
            rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng(0)
        orig, cur = desc, desc
        cur_m = float(margin_fn([cur])[0])
        m0 = cur_m
        budget = word_budget(len(orig.split()), max_frac, min_budget)
        hist, stale, step = [cur_m], 0, 0
        for step in range(1, steps + 1):
            if cur_m < threshold:
                break
            pos, top = self.grad_topk(name, cur, prot_strings, topk)
            if not pos:
                break
            changed = changed_words(orig, cur)
            if len(changed) >= budget:     # budget spent: only re-edit words already changed
                cand_pos = [j for j, (_, s, _) in enumerate(pos) if word_index(cur, s) in changed]
            else:
                cand_pos = list(range(len(pos)))
            if not cand_pos:
                break
            seen, texts = set(), []
            for _ in range(batch * 3):
                if len(texts) >= batch:
                    break
                j = int(rng.choice(cand_pos))
                tid = int(top[j, int(rng.integers(top.shape[1]))])
                if (j, tid) in seen or tid not in self.words:
                    continue
                seen.add((j, tid))
                _, s, e = pos[j]
                t = replace_span(cur, s, e, self.words[tid])
                if t != cur and len(changed_words(orig, t)) <= budget:
                    texts.append(t)
            if not texts:
                break
            ms = np.asarray(margin_fn(texts), dtype=float)
            b = int(np.argmin(ms))
            if ms[b] < cur_m:
                cur, cur_m, stale = texts[b], float(ms[b]), 0
            else:
                stale += 1
            hist.append(cur_m)
            if stale >= patience:
                break
        diff = sorted(changed_words(orig, cur))
        ow, cw = orig.split(), cur.split()
        return {"adv_desc": cur, "margin_orig": m0, "margin_adv": cur_m, "evaded": cur_m < threshold,
                "evaded_p05": cur_m < 0.0, "steps": step, "n_words": len(ow), "budget": budget,
                "n_changed": len(diff), "changes": [[ow[i], cw[i]] for i in diff if i < len(cw)],
                "margin_trace": [round(x, 3) for x in hist]}
