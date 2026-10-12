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


def train_detector(train, val, dcfg: dict, use_context: bool, seed: int, log, field: str = "description"):
    """Returns (model, tok, history). Keeps the epoch with the best val score (AUROC + TPR@5%FPR)."""
    import copy

    import torch

    from ..utils.seed import set_seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = build_model(dcfg["backbone"])
    model.to(device)
    mk = lambda es: Encoded(tok, [detector_text(e, use_context, field) for e in es], [e["label"] for e in es],
                            dcfg["max_len"])
    dtr, dva = mk(train), mk(val)

    pos = sum(e["label"] for e in train)
    w = torch.tensor([len(train) / (2 * (len(train) - pos)), len(train) / (2 * pos)], device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=float(dcfg["lr"]), weight_decay=float(dcfg["weight_decay"]))
    bs, epochs = int(dcfg["batch_size"]), int(dcfg["epochs"])
    steps = epochs * math.ceil(len(dtr) / bs)
    warm = int(float(dcfg["warmup_ratio"]) * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda" and dcfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)

    best, best_state, hist = -1.0, None, []
    yv = [e["label"] for e in val]
    for ep in range(epochs):
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
        score = a + t5
        hist.append({"epoch": ep, "loss": tot / max(1, n), "val_auroc": a, "val_tpr@5": t5, "sec": time.time() - t0})
        log.info("  ep %d loss %.4f val AUROC %.3f TPR@5%% %.3f (%.0fs)", ep, tot / max(1, n), a, t5, time.time() - t0)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, hist, device


def score(model, tok, examples, dcfg, use_context, device, field="description", contexts=None) -> np.ndarray:
    pairs = [detector_text(e, use_context, field, None if contexts is None else contexts[i])
             for i, e in enumerate(examples)]
    ds = Encoded(tok, pairs, [e["label"] for e in examples], dcfg["max_len"])
    return predict(model, tok, ds, int(dcfg["batch_size"]), device)
