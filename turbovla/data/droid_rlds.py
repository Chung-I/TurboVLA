"""Streamed DROID RLDS dataset for TurboVLA.

Ports openpi's ``DroidRldsDataset`` (src/openpi/training/droid_rlds_dataset.py)
to TurboVLA's trainer interface, with three deliberate departures:

- Reads straight from GCS (``gs://gresearch/robotics/droid/1.0.1``) with
  rank-aware file sharding via ``tf.distribute.InputContext``, so a DDP run
  never downloads the 1.8 TB dataset and never reads a shard twice.
- The frame shuffle buffer holds *encoded* JPEG bytes; images are decoded,
  resized and padded to 256x256 after shuffling. openpi shuffles decoded
  frames, which costs ~90 GB RAM at the same buffer size.
- The idle filter is a per-trajectory Python range lookup instead of a
  20M-entry StaticHashTable (minutes of startup, GBs of RAM).

Actions are 8-D: 7 joint-position deltas relative to the state at the chunk's
first frame, plus the absolute gripper position. Chunks are tail-clamped to
the last action like openpi, so every chunk row is valid supervision and the
action mask is all-ones.

Each ``__iter__`` element is a full per-GPU batch in the trainer's tuple
layout ``(samples, instructions, states, action_chunks, action_chunk_masks)``;
use ``DataLoader(dataset, batch_size=None, num_workers=0)``. Images travel as
uint8 ``(B, 2, H, W, 3)``; DINOv3 normalization happens on GPU in the
training shim.
"""

import json
import logging
import time

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

DROID_GCS_DIR = "gs://gresearch/robotics/droid/1.0.1"
# Approximate number of DROID samples surviving the success + idle filters.
APPROX_FILTERED_SAMPLES = 20_000_000


def load_droid_stats(stats_path, stats_key):
    with open(stats_path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)[stats_key]
    action = stats["action"]
    proprio = stats["proprio"]
    return {
        "action_low": np.asarray(action["q01"], dtype=np.float32),
        "action_high": np.asarray(action["q99"], dtype=np.float32),
        "proprio_mean": np.asarray(proprio["mean"], dtype=np.float32),
        "proprio_std": np.asarray(proprio["std"], dtype=np.float32),
    }


class DroidRLDSDataset(IterableDataset):
    def __init__(
        self,
        data_dir=DROID_GCS_DIR,
        stats_path=None,
        stats_key="droid",
        rank=0,
        world_size=1,
        batch_size=32,
        chunk_size=16,
        split="train",
        filter_ranges_path=None,
        shuffle_buffer=150_000,
        interleave_cycle=8,
        num_parallel_calls=4,
        prefetch_batches=6,
        image_size=256,
        seed=42,
        tf_intra_op_threads=2,
        tf_inter_op_threads=2,
        skip_images=False,
    ):
        import tensorflow as tf

        # Must run before TF creates its thread pools; ignore if already set.
        try:
            tf.config.threading.set_intra_op_parallelism_threads(tf_intra_op_threads)
            tf.config.threading.set_inter_op_parallelism_threads(tf_inter_op_threads)
        except RuntimeError:
            pass
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        self.data_dir = data_dir
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size = int(batch_size)
        self.chunk_size = int(chunk_size)
        self.split = split
        self.shuffle_buffer = int(shuffle_buffer)
        self.interleave_cycle = int(interleave_cycle)
        self.num_parallel_calls = int(num_parallel_calls)
        self.prefetch_batches = int(prefetch_batches)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.skip_images = bool(skip_images)

        if stats_path is not None:
            stats = load_droid_stats(stats_path, stats_key)
            self.action_low = stats["action_low"]
            self.action_high = stats["action_high"]
            self.proprio_mean = stats["proprio_mean"]
            self.proprio_std = stats["proprio_std"]
        else:
            # Identity normalization; only acceptable for stats computation.
            self.action_low = None
            self.action_high = None
            self.proprio_mean = None
            self.proprio_std = None

        self._filter_ranges = None
        if filter_ranges_path is not None:
            with open(filter_ranges_path, "r", encoding="utf-8") as handle:
                self._filter_ranges = json.load(handle)
            logger.info("idle filter: %d episodes with non-idle ranges", len(self._filter_ranges))

    def _range_mask(self, folderpath, filepath, traj_len):
        """Per-episode non-idle mask. Episodes absent from the filter dict are
        dropped entirely, matching openpi's default_value=False hash table."""
        n = int(traj_len)
        if self._filter_ranges is None:
            return np.ones(n, dtype=bool)
        key = folderpath.decode("utf-8") + "--" + filepath.decode("utf-8")
        mask = np.zeros(n, dtype=bool)
        for start, end in self._filter_ranges.get(key, ()):
            mask[max(0, int(start)) : min(int(end), n)] = True
        return mask

    def _build_pipeline(self, seed):
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        from dlimp.dataset import _broadcast_metadata_rlds

        chunk_size = self.chunk_size
        image_size = self.image_size
        skip_images = self.skip_images

        builder = tfds.builder_from_directory(builder_dir=self.data_dir)

        read_config = tfds.ReadConfig(
            skip_prefetch=True,
            shuffle_seed=seed,
            num_parallel_calls_for_interleave_files=self.interleave_cycle,
            interleave_cycle_length=self.interleave_cycle,
        )
        if self.world_size > 1:
            read_config.input_context = tf.distribute.InputContext(
                num_input_pipelines=self.world_size,
                input_pipeline_id=self.rank,
            )

        dataset = builder.as_dataset(
            split=self.split,
            shuffle_files=True,
            decoders={"steps": tfds.decode.SkipDecoding()},
            read_config=read_config,
        )

        # Rewrap as a DLataset the same way dlimp's from_rlds does, so we get
        # traj_map/flatten/frame_map while controlling the ReadConfig above.
        dataset.__class__ = type("DLataset", (dl.DLataset, type(dataset)), dict(dl.DLataset.__dict__))
        dataset.is_flattened = False
        dataset = dataset._apply_options()
        dataset = dataset.enumerate().traj_map(_broadcast_metadata_rlds)

        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )
        dataset = dataset.repeat()

        def lookup_mask(folderpath, filepath, traj_len):
            mask = tf.py_function(
                lambda f, p, n: self._range_mask(f.numpy(), p.numpy(), n.numpy()),
                [folderpath, filepath, traj_len],
                tf.bool,
            )
            mask.set_shape([None])
            return mask

        def restructure(traj):
            traj_len = tf.shape(traj["action"])[0]

            state = tf.cast(
                tf.concat(
                    (
                        traj["observation"]["joint_position"],
                        traj["observation"]["gripper_position"],
                    ),
                    axis=-1,
                ),
                tf.float32,
            )
            actions = tf.cast(
                tf.concat(
                    (
                        traj["action_dict"]["joint_position"],
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                ),
                tf.float32,
            )

            # Chunk absolute actions with tail clamping (repeat last action).
            chunk_indices = tf.minimum(
                tf.range(chunk_size)[None, :] + tf.range(traj_len)[:, None],
                traj_len - 1,
            )
            action_chunks = tf.gather(actions, chunk_indices)  # [T, chunk, 8]

            # Delta joint positions relative to the chunk's first-frame state;
            # gripper stays absolute (openpi's make_bool_mask(7, -1)).
            deltas = action_chunks[..., :7] - state[:, None, :7]
            action_chunks = tf.concat((deltas, action_chunks[..., 7:]), axis=-1)

            if self.action_low is not None:
                low = tf.constant(self.action_low)
                high = tf.constant(self.action_high)
                action_chunks = 2.0 * (action_chunks - low) / (high - low + 1e-6) - 1.0
                action_chunks = tf.clip_by_value(action_chunks, -1.0, 1.0)
            if self.proprio_mean is not None:
                state = (state - tf.constant(self.proprio_mean)) / (
                    tf.constant(self.proprio_std) + 1e-6
                )

            # Random exterior camera per trajectory (left of each stereo pair).
            exterior = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist = traj["observation"]["wrist_image_left"]

            # Random instruction among the non-empty annotations.
            candidates = tf.stack(
                (
                    traj["language_instruction"][0],
                    traj["language_instruction_2"][0],
                    traj["language_instruction_3"][0],
                )
            )
            nonempty = tf.reshape(tf.where(tf.strings.length(candidates) > 0), [-1])
            instruction = tf.cond(
                tf.size(nonempty) > 0,
                lambda: candidates[tf.random.shuffle(nonempty)[0]],
                lambda: candidates[0],
            )

            metadata = traj["traj_metadata"]["episode_metadata"]
            passes_filter = lookup_mask(
                metadata["recording_folderpath"][0], metadata["file_path"][0], traj_len
            )

            out = {
                "actions": action_chunks,
                "state": state,
                "prompt": tf.fill([traj_len], instruction),
                "passes_filter": passes_filter,
            }
            if not skip_images:
                out["exterior_image"] = exterior
                out["wrist_image"] = wrist
            return out

        dataset = dataset.traj_map(restructure, self.num_parallel_calls)
        dataset = dataset.flatten(num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.filter(lambda frame: frame["passes_filter"])

        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # Shuffle *encoded* frames, then decode.
        dataset = dataset.shuffle(self.shuffle_buffer, seed=seed)

        if not skip_images:

            def decode_and_resize(frame):
                views = []
                for key in ("exterior_image", "wrist_image"):
                    img = tf.io.decode_image(frame.pop(key), expand_animations=False, dtype=tf.uint8)
                    img = _resize_with_pad_tf(img, image_size, image_size)
                    views.append(img)
                frame["images"] = tf.stack(views)  # [2, H, W, 3] uint8
                return frame

            dataset = dataset.frame_map(decode_and_resize, self.num_parallel_calls)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.with_ram_budget(1)
        dataset = dataset.prefetch(self.prefetch_batches)
        return dataset

    def _to_trainer_batch(self, batch):
        instructions = [p.decode("utf-8") for p in batch["prompt"]]
        states = torch.from_numpy(np.array(batch["state"]))
        action_chunks = torch.from_numpy(np.array(batch["actions"]))
        masks = torch.ones(action_chunks.shape[0], self.chunk_size, dtype=torch.float32)
        if self.skip_images:
            samples = {}
        else:
            samples = {"dinov3": torch.from_numpy(np.ascontiguousarray(batch["images"]))}
        return samples, instructions, states, action_chunks, masks

    def __iter__(self):
        import tensorflow as tf

        # Bind before iterating: module globals may already be torn down when
        # the generator is finalized at interpreter shutdown.
        op_error = tf.errors.OpError

        seed = self.seed + 1009 * self.rank
        attempt = 0
        while True:
            dataset = self._build_pipeline(seed)
            iterator = dataset.as_numpy_iterator()
            try:
                for batch in iterator:
                    attempt = 0
                    yield self._to_trainer_batch(batch)
                return  # not reachable with repeat(), kept for safety
            except op_error as err:
                attempt += 1
                wait = min(60.0 * attempt, 300.0)
                seed += 1
                logger.warning(
                    "DROID GCS pipeline failed (%s: %s); rebuilding with seed %d in %.0fs",
                    type(err).__name__,
                    err,
                    seed,
                    wait,
                )
                time.sleep(wait)

    def __len__(self):
        return APPROX_FILTERED_SAMPLES // max(1, self.world_size)


def _resize_with_pad_tf(image, target_height, target_width):
    """Aspect-preserving resize then center pad, in-graph (uint8 in/out)."""
    import tensorflow as tf

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    scale = tf.minimum(target_height / height, target_width / width)
    new_height = tf.cast(tf.round(height * scale), tf.int32)
    new_width = tf.cast(tf.round(width * scale), tf.int32)
    resized = tf.image.resize(image, (new_height, new_width), method="bilinear", antialias=True)
    resized = tf.cast(tf.clip_by_value(tf.round(resized), 0.0, 255.0), tf.uint8)
    pad_top = (target_height - new_height) // 2
    pad_left = (target_width - new_width) // 2
    return tf.image.pad_to_bounding_box(resized, pad_top, pad_left, target_height, target_width)


def resize_with_pad_np(image, target_height, target_width):
    """NumPy/PIL twin of `_resize_with_pad_tf` for inference-time parity."""
    from PIL import Image

    height, width = image.shape[:2]
    scale = min(target_height / height, target_width / width)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    resized = np.asarray(
        Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR), dtype=np.uint8
    )
    out = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    pad_top = (target_height - new_height) // 2
    pad_left = (target_width - new_width) // 2
    out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    return out


def droid_identity_collate(batch):
    """The dataset yields fully-formed batches; DataLoader must not re-batch."""
    return batch
