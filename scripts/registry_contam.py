"""Step 12: does registry-relative calibration survive a contaminated registry? (own analysis)

Calibration (Steps 5b/9) assumes the server's already-registered tools are honest. A candidate c is flagged iff
    r(c) = m(c) - anchor(c) >= tr,   anchor(c) = max_{t in R(server), name(t) != name(c)} m(t)
(m = clipped logit margin, tr fixed on CLEAN validation registries). If a poisoned tool already sits in R, the anchor
rises and later poisoned tools look "normal for this server". This script trains the FINAL system per seed (round-2
adversarial training, same data rule as calibrate_adv.py) and measures it on the test sets under:

A. Naive contamination: K poisoned test tools (random draws) are already in every server's registry.
   Defences: anchor over APPROVED tools only (tools that passed the calibrated check when they registered, which is
   what "compare only with already-approved tools" means), and a median anchor instead of max.
B. Adaptive anchor ratchet: before the real attack, the attacker registers k stepping-stone tools. Each one is the
   highest-margin description in a pool of real texts (benign, poisoned, black-box and white-box rewrites; margins
   from this model) that still PASSES the current per-server threshold tr + anchor, so approval cannot stop it; each
   accepted plant raises the anchor. Defence: cap the anchor, anchor_eff = min(anchor, cap), with cap = the largest
   validation-server anchor + delta (delta in contam.cap_offsets, logits); the cap's clean false-alarm cost is
   reported too (a low cap undoes part of the calibration on high-shift servers).
The ratchet pool is optimistic for the defender (a white-box attacker can tune a plant's margin continuously), so
B's numbers are an upper bound on robustness for a given k.

Kaggle (1 GPU, ~7 min per seed):  SCRIPT="scripts/registry_contam.py"  ARGS="--tag contam_deb --set 'contam.seeds=[42,43,44]'"
E5: ARGS="--tag contam_e5 --set detector.backbone=intfloat/multilingual-e5-base 'contam.seeds=[42,43,44]'"
(the round-2 adversarial sources are picked by backbone; override with contam.adv_sources=[...])
CPU smoke test of the analysis only (random margins, no training): --set contam.fake_scores=true
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from adv_train import verified_opt_sets
from calibrate_adv import ex_, load_wb
from common import setup_run

from mcptp.data.io import ensure_examples, load_para
from mcptp.eval.metrics import threshold_at_fpr
from mcptp.utils import atomic_write_json

CLIP = math.log((1 - 1e-7) / 1e-7)      # same clip as defense_checks.py / calibrate_adv.py
SRC = {"deb": ["results/wb1", "results/wb1e5", "results/wb3_base_b30_deb", "results/wb3_adv_b30_deb"],
       "e5": ["results/wb1", "results/wb1e5", "results/wb3_base_b30_e5", "results/wb3_adv_b30_e5"]}
DEFAULTS = {"seeds": [42], "adv_sources": None, "train_rounds": ["r0", "r1", "r1b", "r2"],
            "heldout_rounds": ["r1t", "r2t"], "fpr": 0.01, "k_naive": [1, 2, 3], "draws": 5,
            "k_ratchet": [1, 2, 3, 5, 10], "cap_offsets": [0.0, 2.0, 4.0], "fake_scores": False}
DET = ("test_pos", "bb_opt", "wb_heldout_genuine")     # detection sets (higher is better)
FPR = ("test_neg", "test_hardneg")                     # false-alarm sets (lower is better)


# ------------------------------------------------------------------------------------------------------------ data
def build_data(cfg, paths, c, log):
    """Train/val data, round-2 adversarial rewrites, evaluation sets and registered tools."""
    ex = ensure_examples(cfg, paths)
    split_of = {e["id"]: e["split"] for e in ex}
    v = cfg["para"]["version"]
    para = {s: load_para(paths["splits"], v, s) for s in ("train", "val", "test")}
    train = [e for e in ex if e["split"] == "train"] + para["train"]
    val = [e for e in ex if e["split"] == "val"] + para["val"]
    test = [e for e in ex if e["split"] == "test"] + para["test"]
    adv, seen, held, hseen = [], {e["description"] for e in train}, [], set()
    for run in c["adv_sources"]:
        W, gen = load_wb(run)
        if not W:
            log.warning("missing %s", run)
        for r in W:
            if r["n_changed"] == 0:
                continue
            if r["round"] in c["train_rounds"]:
                assert split_of.get(r["seed_id"]) == "train", r["seed_id"]
                if r["adv_desc"] not in seen:
                    seen.add(r["adv_desc"]); adv.append(ex_(r, r["adv_desc"]))
            elif r["round"] in c["heldout_rounds"] and r["key"] in gen and r["adv_desc"] not in hseen:
                hseen.add(r["adv_desc"]); held.append(ex_(r, r["adv_desc"]))
    sets = {"val_neg": [e for e in val if e["label"] == 0],
            "test_neg": [e for e in test if e["label"] == 0],
            "test_hardneg": [e for e in test if e["label"] == 0 and e.get("is_hard_negative")],
            "test_pos": [e for e in test if e["label"] == 1],
            "bb_opt": [e for xs in verified_opt_sets().values() for e in xs],
            "wb_heldout_genuine": held}
    reg = [e for e in ex if e["kind"] == "benign"]          # registered tools, as in calibrate_adv.py
    return train, val, adv, sets, reg


# -------------------------------------------------------------------------------------------------------- analysis
def _stat(vals, stat):
    return None if not vals else (max(vals) if stat == "max" else float(np.median(vals)))


class Analysis:
    """All calibration variants on fixed margins. items: list of (server, name, desc, m)."""

    def __init__(self, items: dict, reg_items: list, fpr: float, cap_offsets=(0.0, 2.0, 4.0)):
        self.items, self.fpr = items, fpr
        self.reg = defaultdict(list)
        for s, n, d, m in reg_items:
            self.reg[s].append((n, d, m))
        self.tg = threshold_at_fpr([x[3] for x in items["val_neg"]], fpr)
        self.tr = {"max": self._fit(lambda it: self.anchor(it, "max"))}
        # a registered tool is APPROVED iff it passes the clean calibrated (max) check against the other tools
        self.ok = {(s, n, d): self.passes(m, self.anchor((s, n, d, m), "max"), self.tr["max"])
                   for s, rows in self.reg.items() for n, d, m in rows}
        self.tr["approved"] = self._fit(lambda it: self.anchor(it, "max", approved=True))
        self.tr["median"] = self._fit(lambda it: self.anchor(it, "median"))
        va = np.array([a for it in items["val_neg"] if (a := self.anchor(it, "max")) is not None])
        self.caps = {f"cap = val max + {d:g}": float(va.max()) + d for d in cap_offsets}

    # anchors -------------------------------------------------------------------------------------------------
    def anchor(self, it, stat="max", approved=False, extra=()):
        s, n, d, _ = it
        vals = [m for n2, d2, m in self.reg.get(s, []) if n2 != n and (not approved or self.ok[(s, n2, d2)])]
        vals += [m for n2, d2, m, ok in extra if n2 != n and d2 != d and (not approved or ok)]
        return _stat(vals, stat)

    def passes(self, m, a, tr):
        return (m < self.tg) if a is None else (m - a < tr)

    def _fit(self, anchor_fn, cap=math.inf):
        r = [m - min(a, cap) for it in self.items["val_neg"] if (a := anchor_fn(it)) is not None for m in [it[3]]]
        return threshold_at_fpr(r, self.fpr)

    def rate(self, name, anchor_fn, tr, cap=math.inf):
        xs = self.items[name]
        if not xs:
            return float("nan")
        return float(np.mean([not self.passes(it[3], None if (a := anchor_fn(it)) is None else min(a, cap), tr)
                              for it in xs]))

    # A. naive contamination ----------------------------------------------------------------------------------
    def naive(self, ks, draws, rng):
        pool = self.items["test_pos"]
        servers = sorted({it[0] for k in DET + FPR for it in self.items[k]})
        rules = {"max": ("max", False), "approved": ("max", True), "median": ("median", False)}
        out = {}
        for rule, (stat, appr) in rules.items():
            row = {"K=0": {k: self.rate(k, lambda it: self.anchor(it, stat, appr), self.tr[rule]) for k in DET + FPR}}
            for K in ks:
                acc = defaultdict(list)
                for _ in range(draws):
                    cont = {}
                    for s in servers:
                        pick = [pool[i] for i in rng.choice(len(pool), size=K, replace=False)]
                        # would this poisoned tool have been approved at registration on server s?
                        cont[s] = [(n, d, m, self.passes(m, self.anchor((s, n, d, m), "max"), self.tr["max"]))
                                   for _, n, d, m in pick]
                    for k in DET + FPR:
                        acc[k].append(self.rate(k, lambda it: self.anchor(it, stat, appr, cont.get(it[0], ())),
                                                self.tr[rule]))
                row[f"K={K}"] = {k: float(np.mean(v)) for k, v in acc.items()}
            out[rule] = row
        # how often does a planted poisoned tool get approved at all? (it must pass to enter an approved-only registry)
        out["poison_approval_rate"] = float(np.mean([self.passes(m, self.anchor((s2, n, d, m), "max"), self.tr["max"])
                                                     for s2 in servers for _, n, d, m in pool]))
        return out

    # B. adaptive ratchet --------------------------------------------------------------------------------------
    def ratchet(self, ks, pool_margins):
        P = np.unique(np.asarray(pool_margins, dtype=float))
        kmax = max(ks)
        variants = {"no cap": math.inf, **self.caps}
        out = {}
        for vname, cap in variants.items():
            tr = self.tr["max"] if cap == math.inf else self._fit(lambda it: self.anchor(it, "max"), cap)
            row = {"tr": tr, "cap": None if cap == math.inf else cap,
                   "clean": {k: self.rate(k, lambda it: self.anchor(it, "max"), tr, cap) for k in DET + FPR}}
            for name in DET:
                flags = np.zeros((len(self.items[name]), kmax + 1), dtype=bool)
                for i, it in enumerate(self.items[name]):
                    a, m = self.anchor(it, "max"), it[3]
                    if a is None:
                        flags[i, :] = m >= self.tg
                        continue
                    for k in range(kmax + 1):
                        flags[i, k] = m - min(a, cap) >= tr
                        j = np.searchsorted(P, min(a, cap) + tr, "left") - 1     # best plant that still passes
                        if j >= 0 and P[j] > a:
                            a = float(P[j])
                row[name] = {f"k={k}": float(flags[:, k].mean()) for k in [0, *ks]} if len(flags) else {}
                if name == "test_pos" and len(flags):
                    need = [int(np.argmin(f)) if (~f).any() else kmax + 1 for f in flags]
                    row["test_pos_median_plants_to_evade"] = float(np.median(need))
            out[vname] = row
        return out


# ------------------------------------------------------------------------------------------------------------ main
def mean_std(dicts):
    """Average every numeric leaf over seeds -> [mean, std]."""
    d0 = dicts[0]
    if isinstance(d0, dict):
        return {k: mean_std([d[k] for d in dicts]) for k in d0}
    if isinstance(d0, (int, float)) and not isinstance(d0, bool):
        a = np.asarray(dicts, dtype=float)
        return [float(np.nanmean(a)), float(np.nanstd(a))]
    return d0


def main():
    """Entry point (see the module docstring for what this script does and how to run it)."""
    cfg, paths, run_dir, log = setup_run("registry_contam", __doc__)
    c = {**DEFAULTS, **(cfg.get("contam") or {})}
    dcfg = cfg["detector"]
    suffix = "e5" if "e5" in dcfg["backbone"].lower() else "deb"
    c["adv_sources"] = c["adv_sources"] or SRC[suffix]
    train, val, adv, sets, reg = build_data(cfg, paths, c, log)
    log.info("backbone %s | sources %s | adversarial rewrites %d | sets %s", dcfg["backbone"], c["adv_sources"],
             len(adv), {k: len(x) for k, x in sets.items()})
    regs = defaultdict(list)
    for e in reg:
        regs[e["server"]].append({"tool_name": e["tool_name"], "description": e["description"]})

    per_seed = {}
    for seed in c["seeds"]:
        rng = np.random.default_rng(seed)
        if c["fake_scores"]:
            M = lambda xs: np.clip(np.array([rng.normal(8 if e["label"] else -8, 3) for e in xs]), -CLIP, CLIP)
        else:
            from mcptp.detector.train import score, train_detector
            log.info("seed %d: training round-2 ADV %s (%d examples)", seed, dcfg["backbone"], len(train) + len(adv))
            m, tok, _, dev = train_detector(train + adv, val, dcfg, False, int(seed), log, registries=dict(regs))
            M = lambda xs: (np.clip(score(m, tok, xs, dcfg, False, dev, margin=True), -CLIP, CLIP)
                            if xs else np.array([]))
        tup = lambda xs, ms: [(e["server"], e["tool_name"], e["description"], float(s)) for e, s in zip(xs, ms)]
        items = {k: tup(x, M(x)) for k, x in sets.items()}
        reg_items = tup(reg, M(reg))
        if not c["fake_scores"]:
            m = None
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        A = Analysis(items, reg_items, c["fpr"], c["cap_offsets"])
        pool = [it[3] for k in items for it in items[k]] + [it[3] for it in reg_items]
        res = {"thresholds": {"global": A.tg, **{f"rel_{k}": v for k, v in A.tr.items()}},
               "caps": A.caps,
               "naive": A.naive(c["k_naive"], c["draws"], rng), "ratchet": A.ratchet(c["k_ratchet"], pool)}
        per_seed[seed] = res
        log.info("seed %d: thresholds %s | naive max K=1 %s | ratchet no-cap test_pos %s", seed, res["thresholds"],
                 res["naive"]["max"]["K=1"], res["ratchet"]["no cap"]["test_pos"])

    S = mean_std(list(per_seed.values()))
    atomic_write_json({"backbone": dcfg["backbone"], "seeds": c["seeds"], "sources": c["adv_sources"],
                       "sizes": {k: len(x) for k, x in sets.items()}, "per_seed": per_seed, "mean_std": S},
                      run_dir / "contam.json")
    f = lambda v: "—" if v is None or (isinstance(v, list) and np.isnan(v[0])) else f"{v[0]:.3f} ± {v[1]:.3f}"
    n = {k: len(x) for k, x in sets.items()}
    L = [f"# Contaminated-registry test of registry-relative calibration ({dcfg['backbone']}, round-2 ADV, "
         f"seeds {c['seeds']}, {c['fpr']:.0%} val-FPR target)", "",
         f"Sizes: {n}. Mean ± std over seeds. FPR columns: lower is better; detection columns: higher is better.", "",
         "## A. Naive contamination (K poisoned tools already registered on every server)", "",
         "| anchor rule | K | test_neg FPR | hard-neg FPR | test_pos | black-box opt | held-out wb genuine |",
         "|---|---|---|---|---|---|---|"]
    for rule in ("max", "approved", "median"):
        for K, row in S["naive"][rule].items():
            L.append(f"| {rule} | {K[2:]} | " + " | ".join(f(row[k]) for k in FPR + DET) + " |")
    L += ["", f"Share of planted poisoned tools that would pass registration (and so enter an approved-only "
          f"registry): {f(S['naive']['poison_approval_rate'])}", "",
          "## B. Adaptive anchor ratchet (k stepping-stone tools registered first, each passing the check)", ""]
    ks = [0, *c["k_ratchet"]]
    for name in DET:
        L += [f"### Detection on {name} (n={n[name]})", "",
              "| anchor | clean test_neg FPR | clean hard-neg FPR | " + " | ".join(f"k={k}" for k in ks) + " |",
              "|---|---|---|" + "---|" * len(ks)]
        for vname, row in S["ratchet"].items():
            L.append(f"| {vname} | {f(row['clean']['test_neg'])} | {f(row['clean']['test_hardneg'])} | "
                     + " | ".join(f(row[name].get(f'k={k}')) for k in ks) + " |")
        L.append("")
    L.append("Median plants needed before a test positive passes (no cap / caps): "
             + ", ".join(f"{v}: {f(r.get('test_pos_median_plants_to_evade'))}" for v, r in S["ratchet"].items()))
    (run_dir / "contam.md").write_text("\n".join(L) + "\n")
    log.info("\n%s", "\n".join(L))


if __name__ == "__main__":
    main()
