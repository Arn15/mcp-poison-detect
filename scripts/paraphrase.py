"""Step 2b (GPU): generate MCPTox-Para paraphrases. Resumable; one JSONL record per (example, k).

Kaggle:  SCRIPT="scripts/paraphrase.py"  ARGS="--tag para_test"
Smoke:   python scripts/paraphrase.py --set para.backend=echo --tag smoke
Resume:  copy the previous run dir into /kaggle/working/outputs/paraphrase/ and pass --resume <dir name>.
Then:    python scripts/freeze_para.py --inputs <run_dir>/generations.jsonl
"""
from __future__ import annotations

import time

from common import setup_run

from mcptp.data.io import ensure_examples
from mcptp.data.mcptox import stable_id
from mcptp.data.paraphrase import build_prompt, check, clean_output, make_backend, pick_style, source_text
from mcptp.utils import JsonlCheckpoint, atomic_write_json


def main():
    cfg, paths, run_dir, log = setup_run("paraphrase", __doc__)
    p = cfg["para"]
    examples = ensure_examples(cfg, paths)
    jobs = []
    for e in sorted(examples, key=lambda x: x["id"]):
        if e["split"] in p["splits"]:
            for k in range(int(p["n_per_example"][e["split"]])):
                jobs.append((e, k))
    ckpt = JsonlCheckpoint(run_dir / "generations.jsonl", id_key="job")
    done = ckpt.done_ids()
    todo = [(e, k) for e, k in jobs if f"{e['id']}#{k}" not in done]
    log.info("%d jobs, %d done, %d to go (splits=%s)", len(jobs), len(done), len(todo), p["splits"])

    backend = make_backend(p)
    bs, t0, n_valid, n_done = int(p["batch_size"]), time.time(), 0, 0
    pending = [(e, k, 0) for e, k in todo]
    while pending:
        batch, pending = pending[:bs], pending[bs:]
        prompts, metas = [], []
        for e, k, attempt in batch:
            style = pick_style(p["styles"], e["id"], k, attempt)
            user, names, values = build_prompt(e, style)
            prompts.append(user)
            metas.append((e, k, attempt, style, names, values))
        outs = backend.generate(prompts, seed=cfg["project"]["seed"] + n_done)
        for (e, k, attempt, style, names, values), raw in zip(metas, outs):
            text = clean_output(raw)
            chk = check(source_text(e), text, names, values, p["min_value_recall"], tuple(p["length_ratio"]))
            if not chk["valid"] and attempt < int(p["max_retries"]):
                pending.append((e, k, attempt + 1))
                continue
            ckpt.append({"job": f"{e['id']}#{k}", "id": "x_" + stable_id(e["id"], str(k)), "orig_id": e["id"],
                         "k": k, "split": e["split"], "label": e["label"], "style": style, "attempt": attempt,
                         "model": backend.name, "source": source_text(e), "description": text, "raw": raw, **chk})
            n_done += 1
            n_valid += chk["valid"]
        if n_done and n_done % 64 < bs:
            rate = n_done / (time.time() - t0)
            log.info("%d/%d written (%.0f%% valid), %.2f/s, eta %.0f min", n_done, len(todo),
                     100 * n_valid / n_done, rate, (len(todo) - n_done) / max(rate, 1e-9) / 60)
            atomic_write_json({"written": n_done, "valid": n_valid}, run_dir / "state.json")
    atomic_write_json({"written": len(ckpt), "complete": True}, run_dir / "state.json")
    log.info("done: %s", run_dir / "generations.jsonl")


if __name__ == "__main__":
    main()
