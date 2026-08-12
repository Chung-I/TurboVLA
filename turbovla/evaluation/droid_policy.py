"""TurboVLA policy adapter for DROID-trained checkpoints.

Mirrors the DROID training pipeline exactly: resize-with-pad to 256x256,
stats-JSON z-scored 8-D state (7 joint positions + gripper), q01/q99 min-max
action normalization, and delta-to-absolute joint-position conversion against
the state the chunk was predicted from. Loads the EMA weights by default,
matching the released LIBERO checkpoints' convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.droid_rlds import load_droid_stats, resize_with_pad_np
from .policy import TurboVLAPolicy, _checkpoint_state_dict, _strip_module_prefix

DROID_ACTION_DIM = 8
DROID_STATE_DIM = 8
DROID_CHUNK_SIZE = 16


class DroidTurboVLAPolicy(TurboVLAPolicy):
    def __init__(
        self,
        ckpt_path: str,
        stats_path: str,
        stats_key: str = "droid",
        use_ema: bool = True,
        **kwargs: Any,
    ) -> None:
        stats = load_droid_stats(stats_path, stats_key)
        self.action_low = stats["action_low"]
        self.action_high = stats["action_high"]
        self.droid_proprio_mean = stats["proprio_mean"]
        self.droid_proprio_std = stats["proprio_std"]
        self.use_ema = bool(use_ema)
        kwargs.setdefault("action_dim", DROID_ACTION_DIM)
        kwargs.setdefault("chunk_size", DROID_CHUNK_SIZE)
        kwargs.setdefault("state_dim", DROID_STATE_DIM)
        kwargs.setdefault("text_padding_length", 32)
        super().__init__(ckpt_path, **kwargs)

    def _load_checkpoint(self) -> None:
        checkpoint = self._checkpoint
        if self.use_ema and isinstance(checkpoint, dict) and "ema_model_state_dict" in checkpoint:
            source_state = checkpoint["ema_model_state_dict"]
            if self.verbose:
                print("[DroidTurboVLAPolicy] loading EMA weights", flush=True)
        else:
            source_state = _checkpoint_state_dict(checkpoint)
            if self.use_ema and self.verbose:
                print(
                    "[DroidTurboVLAPolicy] WARNING: no ema_model_state_dict in checkpoint, "
                    "falling back to raw weights",
                    flush=True,
                )
        source_state = _strip_module_prefix(source_state)
        self.model.load_state_dict(source_state, strict=True)
        if self.verbose:
            print(
                f"[DroidTurboVLAPolicy] strict checkpoint load: {self.ckpt_path} "
                f"({len(source_state)} tensors)",
                flush=True,
            )
        del self._checkpoint

    def _normalize_droid_state(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != DROID_STATE_DIM:
            raise ValueError(f"DROID state must have {DROID_STATE_DIM} dims, got {state.shape}")
        return torch.from_numpy(
            (state - self.droid_proprio_mean) / (self.droid_proprio_std + 1e-6)
        ).float()

    def _build_batch(self, primary_images, wrist_images, states):
        flat_images: list[np.ndarray] = []
        for primary, wrist in zip(primary_images, wrist_images):
            flat_images.extend([primary, wrist])
        dinov3_pixel_values = self.dinov3_processor(flat_images)["pixel_values"]
        batch_size = len(primary_images)
        samples = {
            "dinov3": dinov3_pixel_values.view(batch_size, 2, *dinov3_pixel_values.shape[1:]).to(self.device),
        }
        state_tensors = torch.stack(
            [self._normalize_droid_state(state) for state in states], dim=0
        ).to(self.device)
        return samples, state_tensors

    def denormalize_action_chunk(self, normalized_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(normalized_chunk, dtype=np.float32)
        return 0.5 * (chunk + 1.0) * (self.action_high - self.action_low) + self.action_low

    def predict_droid_action_chunk(
        self,
        exterior_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state: np.ndarray,
    ) -> np.ndarray:
        """Returns an absolute joint-position action chunk (chunk_size, 8).

        `state` is the raw 8-D proprio (7 joint positions + gripper in [0, 1]).
        Images may be any resolution; they are resize-with-padded to 256x256
        with the training-time convention.
        """
        exterior = resize_with_pad_np(np.asarray(exterior_image), 256, 256)
        wrist = resize_with_pad_np(np.asarray(wrist_image), 256, 256)
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        samples, states = self._build_batch([exterior], [wrist], [state])
        samples, states = self._prepare_model_inputs(samples, states)
        with torch.inference_mode():
            pred = self.model([instruction], samples, states)
        normalized = pred.detach().float().cpu().numpy()[0]
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)

        chunk = self.denormalize_action_chunk(normalized)
        # Delta joint positions -> absolute targets; gripper is already absolute.
        chunk[:, :7] = chunk[:, :7] + state[None, :7]
        chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)
        return chunk.astype(np.float32)
