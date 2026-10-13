"""Fine-tuning loop (plain PyTorch, fp16 autocast on GPU) with best-epoch selection on validation."""
from __future__ import annotations

import math
import time

import numpy as np

from ..eval.metrics import auroc, tpr_at_fpr
from .model import build_model, detector_text


class Encoded:
    def __init__(self, tok, pairs, labels, max_len):
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        kw = dict(truncation=True, max_length=max_len, padding=False)
        self.enc = tok(a, b, **kw) if all(x is not None for x in b) else tok(a, **kw)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)


def _batches(ds: Encoded, bs: int, tok, shuffle: bool, rng: np.random.Generator):
    import torch

    idx = np.arange(len(ds))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        feats = [{k: ds.enc[k][j] for k in ds.enc.keys()} for j in sel]
        batch = tok.pad(feats, return_tensors="pt")
        batch["labels"] = torch.tensor([ds.labels[j] for j in sel])
        yield batch


def predict(model, tok, ds: Encoded, bs: int, device) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for batch in _batches(ds, bs * 2, tok, False, None):
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.float()
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def augmented_pairs(train, use_context: bool, field: str, rng, registries: dict | None,
                    context_dropout: float, context_swap: float):
    """Per-epoch training inputs with context augmentation (context-aware model only).

    * context dropout: with prob p, drop the context, so the model can't rely on context alone.
    * context swap: a benign tool with no referenced tools gets random real tools from its own registry as
      context (prob q), so "has context" stops being a poison cue (the D1 pilot showed 0.43 hard-negative FPR).
    """
    pairs = []
    for e in train:
        if not use_context:
            pairs.append(detector_text(e, False, field))
            continue
        ctx = e.get("context") or []
        u = rng.random()
        if ctx and u < context_dropout:
            ctx = []
        elif not ctx and e["label"] == 0 and registries and u < context_swap:
            pool = [t for t in registries.get(e["server"], []) if t["tool_name"] != e["tool_name"]]
            if pool:
                k = int(rng.integers(1, min(3, len(pool)) + 1))
                ctx = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
        pairs.append(detector_text(e, True, field, context_override=ctx))
    return pairs


def train_detector(train, val, dcfg: dict, use_context: bool, seed: int, log, field: str = "description",
                   registries: dict | None = None):
    """Returns (model, tok, history, device). Keeps the epoch with the best val score (AUROC + TPR@5%FPR),
    breaking ties by lower val loss (the D1 pilot's val was saturated, so ties were the norm)."""
    import copy

    import torch

    from ..utils.seed import set_seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = build_model(dcfg["backbone"])
    model.to(device)
    dva = Encoded(tok, [detector_text(e, use_context, field) for e in val], [e["label"] for e in val], dcfg["max_len"])
    ylab = [e["label"] for e in train]
    p_drop, p_swap = float(dcfg.get("context_dropout", 0.0)), float(dcfg.get("context_swap", 0.0))

    pos = sum(e["label"] for e in train)
    w = torch.tensor([len(train) / (2 * (len(train) - pos)), len(train) / (2 * pos)], device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg["lr"]), weight_decay=float(dcfg["weight_decay"]))
    bs, epochs = int(dcfg["batch_size"]), int(dcfg["epochs"])
    steps = epochs * math.ceil(len(train) / bs)
    warm = int(float(dcfg["warmup_ratio"]) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda" and dcfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)

    best, best_state, hist = (-1.0, float("inf")), None, []
    yv = np.array([e["label"] for e in val])
    for ep in range(epochs):
        dtr = Encoded(tok, augmented_pairs(train, use_context, field, rng, registries, p_drop, p_swap), ylab,
                      dcfg["max_len"])
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in _batches(dtr, bs, tok, True, rng):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = lossf(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(labels); n += len(labels)
            if dcfg.get("max_steps") and n >= int(dcfg["max_steps"]) * bs:
                break
        sv = predict(model, tok, dva, bs, device)
        a, t5 = auroc(yv, sv), tpr_at_fpr(yv, sv, 0.05)
        pc = np.clip(sv, 1e-6, 1 - 1e-6)
        vloss = float(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))
        key = (a + t5, -vloss)
        hist.append({"epoch": ep, "loss": tot / max(1, n), "val_auroc": a, "val_tpr@5": t5, "val_loss": vloss,
                     "sec": time.time() - t0})
        log.info("  ep %d loss %.4f val AUROC %.3f TPR@5%% %.3f val-loss %.4f (%.0fs)", ep, tot / max(1, n), a, t5,
                 vloss, time.time() - t0)
        if key > (best[0], -best[1]):
            best = (key[0], vloss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, hist, device


def score(model, tok, examples, dcfg, use_context, device, field="description", contexts=None) -> np.ndarray:
    pairs = [detector_text(e, use_context, field, None if contexts is None else contexts[i])
             for i, e in enumerate(examples)]
    ds = Encoded(tok, pairs, [e["label"] for e in examples], dcfg["max_len"])
    return predict(model, tok, ds, int(dcfg["batch_size"]), device)


def save_detector(model, tok, dcfg, use_context, out_dir, field="description"):
    """Save a fine-tuned detector so the attacker loop can score against it."""
    from pathlib import Path
    import json as _json
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    _json.dump({"backbone": dcfg["backbone"], "max_len": dcfg["max_len"], "use_context": use_context,
                "field": field}, open(out / "detector.json", "w"))
    return out


class LoadedDetector:
    """Score examples with a saved detector. Thin wrapper the attacker loop calls."""

    def __init__(self, path, device=None):
        import json as _json
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.meta = _json.load(open(f"{path}/detector.json"))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dcfg = {"backbone": self.meta["backbone"], "max_len": self.meta["max_len"], "batch_size": 32}

    def score(self, examples):
        return score(self.model, self.tok, examples, self.dcfg, self.meta["use_context"], self.device,
                     field=self.meta.get("field", "description"))
