# PAS-LGT — Physics-informed, Adversarial, Situation-aware Lane Graph Transformer

Long-term highway trajectory prediction for proactive VEC service migration.

**Current status: Phase 1 (baseline Seq2Seq Transformer) implemented.**

Lives in the `bunes/` subfolder of the shared course repo; all paths below are
relative to this folder.

---

## Running on Colab (no local install)

Open [`notebooks/colab_phase1.ipynb`](notebooks/colab_phase1.ipynb) in Google
Colab — via *File → Open notebook → GitHub*, or by prefixing the file's GitHub
URL with `colab.research.google.com/github/`.

Cell 1 clones this repo and `cd`s into `bunes/`. Colab already ships every
Phase 1 dependency (torch, pandas, scipy, pyarrow, tqdm, matplotlib, pytest), so
there is nothing to install and nothing to upload — the dataset is generated in
the notebook. Set the runtime to a T4 GPU and run all cells.

To pick up new commits later, just re-run cell 1.

---

## Quick start (local)

```bash
pip install -r requirements.txt
```

```bash
python scripts/make_dummy_data.py
```

```bash
python -m src.train --config configs/phase1_baseline.yaml
```

```bash
python -m src.evaluate --ckpt runs/phase1_baseline/best.pt --plot
```

To use real data instead of the synthetic generator:

```bash
python -m src.data.preprocess --source ngsim --raw data/raw/ngsim_us101.csv --target-hz 2.0
```

```bash
python -m src.data.preprocess --source highd --raw data/raw/highd --target-hz 2.0
```

Smoke tests:

```bash
pytest -q
```

---

## Conventions fixed in Phase 1

These are contracts the later phases depend on. Changing one means touching
every phase, so they are worth reading once.

| Concern | Decision |
|---|---|
| Coordinates | `x` = longitudinal (direction of travel), `y` = lateral, metres. NGSIM `Local_Y→x`, `Local_X→y`; highD's reverse direction is mirrored. |
| Sampling rate | Raw data resampled to `target_hz` (default **2 Hz**). 10 s history = 20 steps, 30 s horizon = 60 steps. Decoding 300 autoregressive steps at 10 Hz is neither necessary nor stable. |
| Model frame | Every window is translated to the last observed point and rotated to its heading. `origin`/`theta` ride along in each batch so Phase 2 can go back to world coordinates. |
| Model output | An **offset from a constant-velocity anchor**: `pos_t = (t+1)·cv_delta + o_t`. Not per-step displacements — see below. |
| Prior | The anchor is the last observed displacement extrapolated forward, with the head zero-initialised. An untrained model is exactly a CV predictor — this is the "History Message" fusion point that Phase 3 extends with IDM. |
| Evaluation | Always **autoregressive** (`model.rollout`). Teacher-forced validation loss is misleading for this problem. |

## Why the output is an offset, not a displacement

The first working version predicted per-step displacements and recovered
position by `cumsum`. It failed in an instructive way, and the failure is worth
recording because it is the exact phenomenon Phases 2 and 5 exist to fight.

Under teacher forcing, the cheapest way to minimise the loss is to copy the
previous displacement token — speeds change slowly, so "same as last step"
scores almost perfectly. Training loss fell monotonically to ~0.5. But during
free-running rollout the model then integrates its *own* copied error 60 times:
a feedback loop with gain ≈ 1. Symptoms:

* validation ADE oscillating between 7 m and 44 m across consecutive epochs
  while training loss fell smoothly — instability, not overfitting;
* lateral RMSE stable at ~1 m while longitudinal RMSE swung between 18 m and
  71 m — the divergence was purely in the integrated speed;
* the model losing to a constant-velocity baseline (ADE 7.9 m).

Anchoring every step to the prior removes the integrator. An error in `o_t` no
longer propagates into `o_{t+1}`, and if the model still learns to copy, the
degenerate solution is "constant velocity plus a fixed offset" — roughly
baseline quality — instead of exponential divergence. The failure mode becomes
graceful.

Three tests lock this in: `test_untrained_model_is_constant_velocity`,
`test_rollout_does_not_integrate_its_own_error`, and
`test_teacher_forcing_matches_rollout`. Training also prints
`<-- WORSE THAN CONSTANT VELOCITY` on any epoch that loses to the baseline.

## Phase 2 attachment point

`TrajectoryTransformer.rollout(..., step_hook=...)` already accepts a per-step
callback receiving the cumulative agent-frame position and the step index, and
returning a corrected position. Link Projection is:

1. `from_agent_frame(pos, origin, theta)` → world coordinates
2. Shapely / KD-Tree nearest point on the lane centreline
3. `to_agent_frame(snapped, origin, theta)` → returned to the hook

The rollout loop then re-derives the displacement so the token fed back to the
decoder is consistent with the snapped position. No change to the decode loop is
required. `tests/test_smoke.py::test_rollout_hook_is_applied` locks this in.

## Layout

```
configs/          YAML experiment configs (local + Colab variants)
notebooks/        Colab runner notebook
scripts/          dataset generation helpers
src/config.py     typed config (rejects unknown YAML keys)
src/data/         schema, transforms, synthetic sim, NGSIM/highD parsers, windowing
src/models/       positional encoding + the Seq2Seq Transformer
src/metrics.py    ADE / FDE / horizon breakdown / lateral-longitudinal split
src/engine.py     train + autoregressive eval loops, CV baseline
src/train.py      Phase 1 entry point
src/evaluate.py   test-set report + qualitative plots
tests/            smoke tests
```

## Reading the Phase 1 results

`src/evaluate.py` prints the model next to a constant-velocity baseline and
splits the error into longitudinal and lateral RMSE. Expect the baseline to:

* beat constant velocity comfortably on longitudinal error,
* still accumulate several metres of **lateral** error by 30 s, drifting out of
  the starting lane on the example plots.

That lateral drift is the failure mode Phase 2 (Link Projection) exists to fix,
and the `rmse_lat` / `ade@30s` columns are the before/after numbers to quote.

---