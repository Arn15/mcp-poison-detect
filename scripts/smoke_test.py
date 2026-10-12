"""CPU smoke test of the run infrastructure: config -> run dir -> resumable JSONL loop.

Simulates a long job over 50 fake items that "crashes" after --crash-after items.
Re-run with --resume <run_dir_name> and it continues where it stopped.
"""
from __future__ import annotations

import sys

from common import setup_run

from mcptp.utils import JsonlCheckpoint, atomic_write_json


def main():
    crash_after = None
    if "--crash-after" in sys.argv:
        i = sys.argv.index("--crash-after")
        crash_after = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    cfg, paths, run_dir, log = setup_run("smoke", __doc__)
    ckpt = JsonlCheckpoint(run_dir / "items.jsonl")
    done = ckpt.done_ids()
    log.info("resuming with %d items already done", len(done))
    processed = 0
    for item_id in range(50):
        if item_id in done:
            continue
        if crash_after is not None and processed >= crash_after:
            log.warning("simulated crash after %d items", processed)
            sys.exit(1)
        ckpt.append({"id": item_id, "score": item_id / 50})
        processed += 1
        if processed % cfg["checkpoint"]["every_n_items"] == 0:
            atomic_write_json({"n_done": len(done) + processed}, run_dir / "state.json")
    atomic_write_json({"n_done": len(ckpt), "complete": True}, run_dir / "state.json")
    log.info("done: %d items in %s", len(ckpt), run_dir)


if __name__ == "__main__":
    main()
