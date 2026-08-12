#!/usr/bin/env python3
"""Websocket policy server for DROID-trained TurboVLA checkpoints.

Speaks the openpi client protocol (msgpack-numpy frames, metadata sent on
connect, raw observation dict in / raw action dict out), so RoboLab's
pi0_family-style clients can point at it directly. Request keys follow
openpi's DROID convention:

    observation/exterior_image_1_left   uint8 (H, W, 3)
    observation/wrist_image_left        uint8 (H, W, 3)
    observation/joint_position          (7,)
    observation/gripper_position        (1,) or scalar
    prompt                              str

Response: {"actions": (chunk_size, 8) float32} -- absolute joint positions
plus gripper in [0, 1].

Usage:
    python scripts/droid/serve_droid_policy.py \
        --ckpt_path outputs/droid/turbovla_droid_100000.pth \
        --stats_path experiments/droid/configs/droid_stats.json \
        --dinov3_path <local dinov3 dir> --bert_path pretrained/bert-base-uncased \
        --port 8000
"""

import argparse
import asyncio
import logging
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "starvla_runtime"))

from deployment.model_server.tools import msgpack_numpy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_droid_policy")


class DroidPolicyServer:
    def __init__(self, policy, host, port):
        self._policy = policy
        self._host = host
        self._port = port

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server

        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ):
            logger.info("serving on ws://%s:%d", self._host, self._port)
            await asyncio.get_running_loop().create_future()

    async def _handler(self, websocket):
        import websockets.frames

        logger.info("connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"server": "turbovla-droid"}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                action = await asyncio.get_running_loop().run_in_executor(None, self._infer, obs)
                await websocket.send(packer.pack(action))
            except Exception as err:
                if err.__class__.__name__.startswith("ConnectionClosed"):
                    logger.info("connection closed: %s", websocket.remote_address)
                    return
                logger.exception("inference error")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                return

    def _infer(self, obs):
        exterior = np.asarray(obs["observation/exterior_image_1_left"])
        wrist = np.asarray(obs["observation/wrist_image_left"])
        joint_position = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)
        gripper_position = np.asarray(obs["observation/gripper_position"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joint_position, gripper_position[:1]])
        prompt = str(obs.get("prompt", ""))
        chunk = self._policy.predict_droid_action_chunk(exterior, wrist, prompt, state)
        return {"actions": chunk}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--stats_key", type=str, default="droid")
    parser.add_argument("--dinov3_path", type=str, required=True)
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--no_ema", action="store_true", help="Load raw instead of EMA weights")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run one canned inference and exit instead of serving.",
    )
    args = parser.parse_args()

    from turbovla.evaluation.droid_policy import DroidTurboVLAPolicy

    policy = DroidTurboVLAPolicy(
        ckpt_path=args.ckpt_path,
        stats_path=args.stats_path,
        stats_key=args.stats_key,
        dinov3_path=args.dinov3_path,
        bert_path=args.bert_path,
        precision=args.precision,
        use_ema=not args.no_ema,
    )

    if args.self_test:
        rng = np.random.default_rng(0)
        chunk = policy.predict_droid_action_chunk(
            exterior_image=rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            wrist_image=rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            instruction="pick up the banana and put it in the bowl",
            state=np.concatenate([rng.uniform(-1, 1, 7), [0.0]]).astype(np.float32),
        )
        assert chunk.shape == (policy.chunk_size, 8), chunk.shape
        assert np.isfinite(chunk).all()
        print(f"self test OK: chunk shape {chunk.shape}")
        print(chunk[:3])
        return

    DroidPolicyServer(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
