#!/usr/bin/env python3
"""DROID trainer: GCS-streamed RLDS data with pi0.5 optimizer knobs.

Layers on top of ``trainer.py`` the same way ``train_mixed.py`` does, via
monkey patches applied at import time:

- ``pi05`` (imported first) contributes the Pi05AdamW betas, EMA shadow
  weights saved as ``ema_model_state_dict``, and DINOv3 precision control.
- This module swaps the dataset for ``DroidRLDSDataset`` (which yields whole
  per-GPU batches), forces the DataLoader into pass-through mode, normalizes
  uint8 images on GPU, adds wandb logging, checkpoint pruning, and a long
  NCCL timeout (the GCS shuffle buffer takes minutes to fill, during which
  rank 0 blocks in allreduce).

DROID-appropriate defaults (action_dim=8, chunk_size=16, 100k steps, warmup
1k, constant post-warmup LR) are injected as argv defaults, so explicit CLI
flags still win.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import functools
import glob
import os
import re
import sys
import threading
import time

import torch

from . import pi05, trainer
from ..data.droid_rlds import DROID_GCS_DIR, DroidRLDSDataset

_DROID_ARG_DEFAULTS = {
    "--dataset_dir": DROID_GCS_DIR,
    "--action_dim": "8",
    "--chunk_size": "16",
    "--state_dim": "8",
    "--batch_size": "32",
    "--grad_accum_steps": "4",
    "--max_steps": "100000",
    "--warmup_steps": "1000",
    "--lr": "5e-5",
    "--min_lr_ratio": "1.0",
    "--num_workers": "0",
    # 60k/rank (120k aggregate) rather than openpi's 250k: cml18 has a 1GbE
    # NIC, and every pipeline rebuild re-pulls the whole buffer at max rate --
    # large buffers turn each recovery into a link-saturating 10GB burst.
    "--shuffle_buffer": "60000",
    "--text_padding_length": "32",
    "--text_layout_path": "",
    "--save_steps": "1000",
}


def parse_args_droid():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dinov3_precision",
        type=str,
        default="bf16_autocast",
        choices=["fp32", "bf16", "bf16_autocast"],
    )
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--stats_key", type=str, default="droid")
    parser.add_argument("--filter_ranges_path", type=str, default=None)
    parser.add_argument("--interleave_cycle", type=int, default=8)
    parser.add_argument("--prefetch_batches", type=int, default=6)
    parser.add_argument("--keep_checkpoint_every", type=int, default=20000)
    parser.add_argument("--keep_last_checkpoints", type=int, default=2)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "turbovla-droid"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"))
    known, remaining = parser.parse_known_args()

    for flag, value in _DROID_ARG_DEFAULTS.items():
        if flag not in remaining:
            remaining += [flag, value]

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = pi05._ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    for key, value in vars(known).items():
        setattr(args, key, value)
    return args


trainer._ACTIVE_TRAIN_ARGS = None

_LAST_STEP_TIME = time.monotonic()


def _stall_stack_dumper(stale_after_s=600.0, check_every_s=60.0):
    """Dump every thread's stack to stderr when optimizer steps stop.

    The training loop can starve in places the dataloader watchdog cannot see
    (DDP collectives, GCS-stalled TF threads). This makes any silent hang
    self-diagnosing in the log.
    """
    rank = os.environ.get("RANK", "?")
    while True:
        time.sleep(check_every_s)
        stale = time.monotonic() - _LAST_STEP_TIME
        if stale > stale_after_s:
            print(
                f"[stall-debug rank {rank}] no optimizer step for {stale:.0f}s; "
                "dumping all thread stacks:",
                file=sys.stderr,
                flush=True,
            )
            faulthandler.dump_traceback(file=sys.stderr)
            sys.stderr.flush()
            time.sleep(540.0)  # at most ~1 dump per stall window


def parse_args_and_record():
    args = parse_args_droid()
    trainer._ACTIVE_TRAIN_ARGS = args

    threading.Thread(target=_stall_stack_dumper, daemon=True, name="stall-debug").start()

    if int(os.environ.get("RANK", "0")) == 0 and args.wandb_mode != "disabled":
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
            resume="allow",
        )
    return args


class TrainerDroidRLDSDataset(DroidRLDSDataset):
    """Adapts the trainer's LiberoRLDSDataset construction call."""

    def __init__(
        self,
        dataset_dir,
        LOCAL_DINOV3_PATH=None,
        rank=0,
        world_size=1,
        chunk_size=16,
        split="train",
        shuffle_buffer=150_000,
        shuffle_steps_within_episode=False,
        step_mix_buffer_size=0,
        seed=42,
        local_files_only=True,
        expected_image_size=256,
    ):
        active_args = trainer._ACTIVE_TRAIN_ARGS
        super().__init__(
            data_dir=dataset_dir,
            stats_path=active_args.stats_path,
            stats_key=active_args.stats_key,
            rank=rank,
            world_size=world_size,
            batch_size=active_args.batch_size,
            chunk_size=chunk_size,
            split=split,
            filter_ranges_path=active_args.filter_ranges_path,
            shuffle_buffer=shuffle_buffer,
            interleave_cycle=active_args.interleave_cycle,
            prefetch_batches=active_args.prefetch_batches,
            image_size=expected_image_size,
            seed=seed,
        )


def droid_passthrough_dataloader(
    dataset,
    batch_size=None,
    sampler=None,
    collate_fn=None,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    **kwargs,
):
    """The dataset already emits whole batches; the DataLoader must not re-batch."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=pin_memory,
    )


_IMAGE_NORM_CACHE = {}


def _image_norm(device):
    stats = _IMAGE_NORM_CACHE.get(device)
    if stats is None:
        from transformers import AutoImageProcessor

        args = trainer._ACTIVE_TRAIN_ARGS
        processor = AutoImageProcessor.from_pretrained(
            args.dinov3_path, local_files_only=not args.allow_hf_download
        )
        mean = torch.tensor(processor.image_mean, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(processor.image_std, device=device).view(1, 1, 3, 1, 1)
        stats = (mean, std)
        _IMAGE_NORM_CACHE[device] = stats
    return stats


def move_samples_to_device_droid(samples, device):
    out = {}
    for key, value in samples.items():
        value = value.to(device, non_blocking=True)
        if value.dtype == torch.uint8:
            # (B, views, H, W, 3) uint8 -> (B, views, 3, H, W) normalized float
            mean, std = _image_norm(device)
            value = value.permute(0, 1, 4, 2, 3).float().div_(255.0).sub_(mean).div_(std)
        out[key] = value
    return out


_ORIGINAL_REDUCE_MEAN = trainer.reduce_mean
_WANDB_LAST_LOG_TIME = None


def reduce_mean_with_wandb(value, device, is_distributed, world_size):
    global _WANDB_LAST_LOG_TIME, _LAST_STEP_TIME
    global_loss = _ORIGINAL_REDUCE_MEAN(value, device, is_distributed, world_size)
    _LAST_STEP_TIME = time.monotonic()
    if int(os.environ.get("RANK", "0")) == 0:
        import wandb

        if wandb.run is not None:
            now = time.monotonic()
            log = {"train/loss": global_loss}
            optimizer = pi05._ACTIVE_OPTIMIZER
            if optimizer is not None:
                log["train/lr"] = optimizer.param_groups[0]["lr"]
            if _WANDB_LAST_LOG_TIME is not None:
                args = trainer._ACTIVE_TRAIN_ARGS
                step_time = now - _WANDB_LAST_LOG_TIME
                log["train/steps_per_s"] = 1.0 / max(step_time, 1e-9)
                log["train/samples_per_s"] = (
                    args.batch_size * args.grad_accum_steps * world_size / max(step_time, 1e-9)
                )
            _WANDB_LAST_LOG_TIME = now
            wandb.log(log)
    return global_loss


_PI05_TORCH_SAVE = trainer.torch.save


def _prune_checkpoints(ckpt_dir, prefix, keep_every, keep_last):
    ckpts = []
    for path in glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth")):
        match = re.search(rf"{re.escape(prefix)}_(\d+)\.pth$", os.path.basename(path))
        if match:
            ckpts.append((int(match.group(1)), path))
    ckpts.sort()
    latest = {path for _, path in ckpts[-keep_last:]}
    for step, path in ckpts:
        if path in latest:
            continue
        if keep_every > 0 and step % keep_every == 0:
            continue
        try:
            os.remove(path)
            print(f"pruned checkpoint: {path}")
        except OSError as err:
            print(f"failed to prune {path}: {err}")


def torch_save_with_pruning(obj, *save_args, **save_kwargs):
    result = _PI05_TORCH_SAVE(obj, *save_args, **save_kwargs)
    args = trainer._ACTIVE_TRAIN_ARGS
    if (
        args is not None
        and isinstance(obj, dict)
        and "model_state_dict" in obj
        and int(os.environ.get("RANK", "0")) == 0
    ):
        _prune_checkpoints(
            args.checkpoint_dir,
            args.checkpoint_prefix,
            args.keep_checkpoint_every,
            args.keep_last_checkpoints,
        )
    return result


_ORIGINAL_INIT_PROCESS_GROUP = trainer.dist.init_process_group
trainer.dist.init_process_group = functools.partial(
    _ORIGINAL_INIT_PROCESS_GROUP, timeout=datetime.timedelta(hours=2)
)

trainer.parse_args = parse_args_and_record
trainer.LiberoRLDSDataset = TrainerDroidRLDSDataset
trainer.DataLoader = droid_passthrough_dataloader
trainer.move_samples_to_device = move_samples_to_device_droid
trainer.reduce_mean = reduce_mean_with_wandb
trainer.torch.save = torch_save_with_pruning


def main():
    trainer.train_model()


if __name__ == "__main__":
    main()
