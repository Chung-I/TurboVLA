#!/usr/bin/env python3
"""Compute DROID normalization stats by streaming a sample from GCS.

Streams the same filtered pipeline the trainer uses (success + idle filters,
delta joint-position actions, absolute gripper) with images skipped, and
aggregates per-dim mean/std/min/max plus q01/q99 quantiles for actions and
proprio. Output is a LIBERO-style stats JSON consumed by both the trainer
and the eval policy.

Roughly 1M frames stream in ~20-40 min depending on bandwidth (the image
bytes still traverse the network even though they are never decoded).

Usage:
    python scripts/droid/compute_droid_stats.py \
        --filter_ranges_path /path/to/droid_sample_ranges_v1_0_1.json \
        --output experiments/droid/configs/droid_stats.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from turbovla.data.droid_rlds import DROID_GCS_DIR, DroidRLDSDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DROID_GCS_DIR)
    parser.add_argument("--filter_ranges_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/droid/configs/droid_stats.json")
    parser.add_argument("--stats_key", type=str, default="droid")
    parser.add_argument("--max_frames", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--quantile_sample_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.filter_ranges_path is None:
        print(
            "WARNING: no --filter_ranges_path given; stats will include idle frames "
            "and will NOT match the filtered training distribution.",
            flush=True,
        )

    dataset = DroidRLDSDataset(
        data_dir=args.data_dir,
        stats_path=None,  # raw values; identity normalization
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        filter_ranges_path=args.filter_ranges_path,
        shuffle_buffer=1,  # aggregation does not need shuffling
        seed=args.seed,
        skip_images=True,
    )

    rng = np.random.default_rng(args.seed)
    action_dim = None
    action_sum = None
    action_sumsq = None
    action_min = None
    action_max = None
    action_samples = []
    proprio_sum = None
    proprio_sumsq = None
    proprio_min = None
    proprio_max = None
    proprio_samples = []
    n_frames = 0
    n_action_rows = 0
    start = time.monotonic()

    for _, _, states, action_chunks, _ in dataset:
        actions = action_chunks.numpy().reshape(-1, action_chunks.shape[-1]).astype(np.float64)
        states_np = states.numpy().astype(np.float64)
        if action_dim is None:
            action_dim = actions.shape[-1]
            action_sum = np.zeros(action_dim)
            action_sumsq = np.zeros(action_dim)
            action_min = np.full(action_dim, np.inf)
            action_max = np.full(action_dim, -np.inf)
            proprio_sum = np.zeros(states_np.shape[-1])
            proprio_sumsq = np.zeros(states_np.shape[-1])
            proprio_min = np.full(states_np.shape[-1], np.inf)
            proprio_max = np.full(states_np.shape[-1], -np.inf)

        action_sum += actions.sum(axis=0)
        action_sumsq += (actions**2).sum(axis=0)
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        n_action_rows += actions.shape[0]

        proprio_sum += states_np.sum(axis=0)
        proprio_sumsq += (states_np**2).sum(axis=0)
        proprio_min = np.minimum(proprio_min, states_np.min(axis=0))
        proprio_max = np.maximum(proprio_max, states_np.max(axis=0))

        keep = rng.random(actions.shape[0]) < args.quantile_sample_fraction
        if keep.any():
            action_samples.append(actions[keep].astype(np.float32))
        keep_s = rng.random(states_np.shape[0]) < args.quantile_sample_fraction
        if keep_s.any():
            proprio_samples.append(states_np[keep_s].astype(np.float32))

        n_frames += states_np.shape[0]
        if n_frames % (args.batch_size * 50) < args.batch_size:
            rate = n_frames / max(time.monotonic() - start, 1e-9)
            print(f"{n_frames}/{args.max_frames} frames ({rate:.0f} frames/s)", flush=True)
        if n_frames >= args.max_frames:
            break

    action_samples = np.concatenate(action_samples, axis=0)
    proprio_samples = np.concatenate(proprio_samples, axis=0)
    action_mean = action_sum / n_action_rows
    action_std = np.sqrt(np.maximum(action_sumsq / n_action_rows - action_mean**2, 0.0))
    proprio_mean = proprio_sum / n_frames
    proprio_std = np.sqrt(np.maximum(proprio_sumsq / n_frames - proprio_mean**2, 0.0))

    stats = {
        args.stats_key: {
            "num_frames": int(n_frames),
            "num_action_rows": int(n_action_rows),
            "num_quantile_samples": int(action_samples.shape[0]),
            "action": {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "min": action_min.tolist(),
                "max": action_max.tolist(),
                "q01": np.quantile(action_samples, 0.01, axis=0).astype(np.float64).tolist(),
                "q99": np.quantile(action_samples, 0.99, axis=0).astype(np.float64).tolist(),
            },
            "proprio": {
                "mean": proprio_mean.tolist(),
                "std": proprio_std.tolist(),
                "min": proprio_min.tolist(),
                "max": proprio_max.tolist(),
                "q01": np.quantile(proprio_samples, 0.01, axis=0).astype(np.float64).tolist(),
                "q99": np.quantile(proprio_samples, 0.99, axis=0).astype(np.float64).tolist(),
            },
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"wrote {args.output}", flush=True)

    np.set_printoptions(precision=4, suppress=True)
    print("action q01:", np.asarray(stats[args.stats_key]["action"]["q01"]), flush=True)
    print("action q99:", np.asarray(stats[args.stats_key]["action"]["q99"]), flush=True)
    print("proprio mean:", np.asarray(stats[args.stats_key]["proprio"]["mean"]), flush=True)
    print("proprio std:", np.asarray(stats[args.stats_key]["proprio"]["std"]), flush=True)


if __name__ == "__main__":
    main()
