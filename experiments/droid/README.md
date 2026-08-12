# DROID training recipe

Trains TurboVLA on the [DROID dataset](https://droid-dataset.github.io/) streamed
directly from GCS — no local copy of the 1.8 TB RLDS dataset is needed. The recipe
follows openpi's `pi05_full_droid_finetune` data conventions so the resulting
policy drops into DROID-jointpos simulator evaluations (e.g. RoboLab).

## Conventions

| | value |
|---|---|
| data | `gs://gresearch/robotics/droid/1.0.1` (RLDS), success + non-idle frames only |
| actions | 8-D: 7 joint-position deltas vs the chunk's first-frame state + absolute gripper |
| state | 8-D: 7 joint positions + gripper position |
| chunk | 16 @ 15 Hz |
| views | one exterior camera (randomly `exterior_image_1_left` or `_2_left` per trajectory) + `wrist_image_left`, resize-with-pad to 256x256 |
| normalization | actions q01/q99 -> [-1, 1] clipped; state z-scored (`configs/droid_stats.json`) |
| optimizer | Pi05AdamW (betas 0.9/0.95) + EMA 0.999, L1 loss, LR 5e-5 (1k warmup, then constant) |
| batch | global 256 = 2 GPUs x 32 x grad-accum 4, 100k steps (~1.3 epochs) |

## Setup

Dependencies beyond the base package: `tensorflow-cpu`, `tensorflow-datasets`,
`dlimp` (pinned to the same revision openpi uses), and `wandb`:

```bash
pip install tensorflow-cpu tensorflow-datasets wandb \
  "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
```

Assets (default root `/tmp2/chungyili/droid`, override with `DROID_ROOT`):

- `dinov3-vitb16/` — local snapshot of `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated on HF)
- `bert-base-uncased/`
- `groundingdino_swint_ogc.pth` — fusion-layer init, from
  `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`
- `droid_sample_ranges_v1_0_1.json` — openpi's non-idle ranges, from
  `https://storage.googleapis.com/openpi-assets/droid/droid_sample_ranges_v1_0_1.json`

## Normalization stats (run once, before training)

```bash
python scripts/droid/compute_droid_stats.py \
  --filter_ranges_path $DROID_ROOT/droid_sample_ranges_v1_0_1.json \
  --output experiments/droid/configs/droid_stats.json
```

Streams ~1M filtered frames (no image decode). Sanity: the joint-delta q99 values
should be on the order of 0.03–0.3 rad; gripper min/max should be ~[0, 1].

## Train

```bash
scripts/droid/train_cml18.sh smoke   # 2k steps, verifies the whole pipeline
scripts/droid/train_cml18.sh full    # 100k steps
```

Notes:

- The shuffle buffer (150k frames/rank) takes several minutes to fill before the
  first step; the NCCL timeout is raised to 2 h to tolerate this and GCS retries.
- Transient GCS failures rebuild the pipeline with a bumped seed automatically.
- Checkpoints are pruned to the last 2 plus every 20k steps (`--keep_checkpoint_every`).
- Training logs to wandb (`WANDB_PROJECT`, default `turbovla-droid`).

## Serve / evaluate

```bash
python scripts/droid/serve_droid_policy.py \
  --ckpt_path <checkpoint.pth> \
  --stats_path experiments/droid/configs/droid_stats.json \
  --dinov3_path <dinov3 dir> --bert_path <bert dir> \
  --port 8000
```

The server speaks the openpi websocket protocol with openpi's DROID observation
keys and returns `{"actions": (16, 8)}` absolute joint positions + gripper in
[0, 1] — directly compatible with RoboLab's `droid_jointpos` environments
(execute ~8 of 16 actions open-loop at 15 Hz). `--self_test` runs one canned
inference without a simulator.
