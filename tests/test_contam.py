"""Unit tests for the contaminated-registry analysis (scripts/registry_contam.py), CPU only, synthetic margins."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from registry_contam import Analysis, mean_std  # noqa: E402


def _items(rng):
    """3 servers x 6 benign registered tools (margins ~ -10), poisoned candidates at +8, benign candidates at -10."""
    servers = ["s0", "s1", "s2"]
    reg = [(s, f"{s}_t{i}", f"{s} benign {i}", float(rng.normal(-10, 0.5))) for s in servers for i in range(6)]
    neg = lambda tag, n: [(s, f"{s}_{tag}{i}", f"{s} {tag} {i}", float(rng.normal(-10, 0.5)))
                          for s in servers for i in range(n)]
    pos = lambda tag, n: [(s, f"{s}_{tag}{i}", f"{s} {tag} {i}", float(rng.normal(8, 1.0))) for s in servers for i in range(n)]
    items = {"val_neg": neg("v", 30), "test_neg": neg("n", 20), "test_hardneg": neg("h", 10), "test_pos": pos("p", 20),
             "bb_opt": pos("b", 10), "wb_heldout_genuine": pos("w", 5)}
    return items, reg


def test_clean_calibration_detects_and_keeps_fpr_low():
    items, reg = _items(np.random.default_rng(0))
    A = Analysis(items, reg, 0.01)
    out = A.naive([1], draws=2, rng=np.random.default_rng(1))
    assert out["max"]["K=0"]["test_pos"] == 1.0
    assert out["max"]["K=0"]["test_neg"] <= 0.1


def test_poisoned_registry_breaks_max_but_not_approved_only():
    items, reg = _items(np.random.default_rng(0))
    A = Analysis(items, reg, 0.01)
    out = A.naive([2], draws=3, rng=np.random.default_rng(1))
    assert out["max"]["K=2"]["test_pos"] < 0.6          # contaminants raise the anchor to poisoned level
    assert out["approved"]["K=2"]["test_pos"] == 1.0     # contaminants would have been rejected at registration
    assert out["poison_approval_rate"] == 0.0


def test_ratchet_moves_anchor_at_most_tr_per_plant():
    items, reg = _items(np.random.default_rng(0))
    A = Analysis(items, reg, 0.01)
    pool = np.linspace(-12, 12, 2000)                      # dense pool: the attacker can pick almost any margin
    out = A.ratchet([1, 2, 3], pool)
    nocap = out["no cap"]["test_pos"]
    assert nocap["k=0"] == nocap["k=3"] == 1.0           # +8 vs ~-10: 3 plants of <= tr each cannot close the gap
    assert A.tr["max"] < 2.0


def test_mean_std_averages_numeric_leaves():
    m = mean_std([{"a": 1.0, "b": {"c": 2.0}, "s": "x"}, {"a": 3.0, "b": {"c": 4.0}, "s": "x"}])
    assert m["a"] == [2.0, 1.0] and m["b"]["c"] == [3.0, 1.0] and m["s"] == "x"
