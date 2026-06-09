from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Any

import numpy as np

from dreamer.veorl_metaworld import VeoRLMetaWorld


def _encode_array(x: Any) -> dict[str, Any]:
    arr = np.asarray(x)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _encode_obs(obs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            out[key] = _encode_array(value)
        else:
            out[key] = value
    return out


def _write(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve VeoRL-style Meta-World over stdio JSON lines.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--camera_name", type=str, default="corner2")
    parser.add_argument("--render_width", type=int, default=128)
    parser.add_argument("--render_height", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--mujoco_gl", type=str, default="egl")
    parser.add_argument("--camera_pos", type=float, nargs=3, default=None)
    parser.add_argument("--camera_fovy", type=float, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    env = VeoRLMetaWorld(
        args.env,
        seed=args.seed,
        action_repeat=args.action_repeat,
        size=(args.render_width, args.render_height),
        camera=args.camera_name,
        camera_pos=tuple(args.camera_pos) if args.camera_pos is not None else None,
        camera_fovy=args.camera_fovy,
    )

    init_payload = {
        "chosen_env_key": getattr(env, "_chosen_env_key", None),
        "action_low": _encode_array(env.action_space.low),
        "action_high": _encode_array(env.action_space.high),
        "action_shape": list(env.action_space.shape),
        "camera_name": args.camera_name,
        "camera_pos": list(args.camera_pos) if args.camera_pos is not None else None,
        "camera_fovy": args.camera_fovy,
        "render_width": args.render_width,
        "render_height": args.render_height,
    }
    _write({"ok": True, "type": "init", "payload": init_payload})

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "reset":
                obs = env.reset()
                _write({"ok": True, "type": "reset", "obs": _encode_obs(obs)})
            elif cmd == "step":
                action = np.asarray(req["action"], dtype=np.float32)
                obs, reward, done, info = env.step(action)
                _write(
                    {
                        "ok": True,
                        "type": "step",
                        "obs": _encode_obs(obs),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                    }
                )
            elif cmd == "close":
                env.close()
                _write({"ok": True, "type": "close"})
                return
            else:
                _write({"ok": False, "error": f"unknown cmd: {cmd}"})
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
