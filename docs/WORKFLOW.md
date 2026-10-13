# Workflow

## Where things run
| Task | Where |
|---|---|
| Writing code, CPU tests, git | Local (VS Code / Claude Code) |
| Detector fine-tuning, baselines | Kaggle T4 (or Colab) |
| Attacker loop, agent ASR eval (vLLM) | Kaggle 2×T4 or Colab A100, `GPU_DEPS=True` |
| Report, analysis | Claude project |

## Loop for every experiment
1. Write or modify a script locally; run `make test` (and a tiny `--set data.max_cases=5` run if possible).
2. `git commit && git push`.
3. In `notebooks/kaggle_runner.ipynb` set `SCRIPT` / `ARGS`, then **Save Version → Save & Run All** (runs in the background, up to 12 h).
4. Outputs land in the notebook's output dataset. Attach it as an input to the next run if needed.
5. Copy final metrics into `results/` (tracked, small JSON/CSV only) for the report.

## Long jobs / resuming
Jobs write one JSONL record per finished item (`mcptp.utils.JsonlCheckpoint`). If a session dies, copy the
previous output dir back in and re-run with `--resume <run_dir_name>`; finished items are skipped.

## One-time setup
- **GitHub**: create an empty repo `mcp-poison-detect` (private until the deadline is fine), then
  `git remote add origin https://github.com/Arn15/mcp-poison-detect.git && git push -u origin main`.
- **Kaggle**: phone-verify the account (needed for GPU + Internet). Add-ons → Secrets: `HF_TOKEN`, `GH_TOKEN`
  (fine-grained PAT, read-only contents on this repo) if private.
- **Hugging Face**: accept the Llama-3.1 license on its model page; create a read token.
- **Colab**: key icon → add the same secrets; enable notebook access.
- **Budget**: Kaggle ~30 GPU-h/week resets weekly. Log GPU hours per run in `docs/compute_log.md`.

## Rules
- Never commit API keys, model weights or raw data (`.gitignore` enforces it).
- Never tune on `data/splits/test*`. The test manifest is frozen in Step 2 with a SHA-256 hash.
- Every reported number must come from a run dir with a saved `config.yaml` + git commit.

## Step 4 runs (attacker)
All use the Kaggle runner, GPU_DEPS=False (transformers backend; vLLM optional later).
1. Checkpoints + verify (one run, ~20 min):
   - `train_detector` with `--set detector.save_ckpt=true 'detector.only=[B3_plain+para,ours_context+para]' --tag ckpt`
     -> saves ckpt_* under the run dir. Save Version so the output is reusable as an input Dataset.
2. Agent-harness check (~5 min): `verify_agent --tag verify --set attacker.max_seeds=60`
   -> want low MAE and positive correlation vs MCPTox ASR before trusting the loop.
3. Attack round 0 (~2-3 h, resumable): `run_attack --tag r0 --set attacker.max_seeds=220`
   (detector_ckpt left null: round 0 optimizes ASR + naturalness only; detector-in-loop is Step 5).
4. Score (~5 min): `score_attacks --tag r0 --set score.attacks=/kaggle/input/<atk>/.../attacks.jsonl
   'score.ckpts=[/kaggle/input/<ck>/.../ckpt_B3_plain+para,/kaggle/input/<ck>/.../ckpt_ours_context+para]'`
