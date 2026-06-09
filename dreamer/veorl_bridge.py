from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload["data_b64"])
    arr = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
    return arr.reshape(payload["shape"])


def _decode_obs(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and {"shape", "dtype", "data_b64"} <= set(value.keys()):
            out[key] = _decode_array(value)
        else:
            out[key] = value
    return out


@dataclass
class _ActionSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]


class VeoRLBridgeEnv:
    """
    Spawn the legacy VeoRL-compatible Meta-World sampling stack in a separate
    Python environment and communicate over stdio JSON lines.
    """

    def __init__(
        self,
        task_name: str,
        *,
        camera_name: str = "corner2",
        camera_pos: tuple[float, float, float] | None = None,
        camera_fovy: float | None = None,
        render_width: int = 128,
        render_height: int = 128,
        seed: int = 0,
        action_repeat: int = 1,
        python_bin: str | os.PathLike[str] = ".conda-veorl/bin/python",
        server_script: str | os.PathLike[str] = "scripts/veorl_metaworld_server.py",
        mujoco_gl: str = "egl",
        extra_env: Optional[dict[str, str]] = None,
        cwd: str | os.PathLike[str] | None = None,
    ):
        self._python_bin = Path(python_bin)
        self._server_script = Path(server_script)
        self._cwd = Path(cwd or Path(__file__).resolve().parents[1])
        self._proc: subprocess.Popen[str] | None = None
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] | None = None

        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", mujoco_gl)
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        mujoco210 = str(Path.home() / ".mujoco" / "mujoco210")
        env.setdefault("MUJOCO_PY_MUJOCO_PATH", mujoco210)
        repo_root = str(self._cwd)
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_parts = [p for p in existing_pythonpath.split(":") if p]
        if repo_root not in pythonpath_parts:
            pythonpath_parts.insert(0, repo_root)
        env["PYTHONPATH"] = ":".join(pythonpath_parts)
        ld_parts = [env.get("LD_LIBRARY_PATH", "")]
        for part in (f"{mujoco210}/bin", "/usr/lib/nvidia"):
            if part and part not in ld_parts[0]:
                ld_parts.append(part)
        env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_parts if p)
        if extra_env:
            env.update(extra_env)

        cmd = [
            str(self._python_bin),
            str(self._server_script),
            "--env",
            task_name,
            "--camera_name",
            camera_name,
            *(
                ["--camera_pos", *[str(float(x)) for x in camera_pos]]
                if camera_pos is not None
                else []
            ),
            *(
                ["--camera_fovy", str(float(camera_fovy))]
                if camera_fovy is not None
                else []
            ),
            "--render_width",
            str(int(render_width)),
            "--render_height",
            str(int(render_height)),
            "--seed",
            str(int(seed)),
            "--action_repeat",
            str(int(action_repeat)),
            "--mujoco_gl",
            mujoco_gl,
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        init_msg = self._read_message(expect_type="init")
        payload = init_msg["payload"]
        self._chosen_env_key = payload.get("chosen_env_key")
        self.action_space = _ActionSpace(
            low=_decode_array(payload["action_low"]).astype(np.float32),
            high=_decode_array(payload["action_high"]).astype(np.float32),
            shape=tuple(payload["action_shape"]),
        )

    def _read_message(self, *, expect_type: str | None = None) -> dict[str, Any]:
        assert self._proc is not None
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            stderr_text = ""
            if self._proc.stderr is not None:
                try:
                    stderr_text = self._proc.stderr.read()
                except Exception:
                    stderr_text = ""
            raise RuntimeError(
                "Legacy VeoRL bridge terminated before replying."
                + (f"\n[stderr]\n{stderr_text}" if stderr_text else "")
            )
        msg = json.loads(line)
        if not msg.get("ok", False):
            raise RuntimeError(f"Legacy VeoRL bridge error: {msg}")
        if expect_type is not None and msg.get("type") != expect_type:
            raise RuntimeError(f"Legacy VeoRL bridge expected type={expect_type!r}, got {msg.get('type')!r}")
        return msg

    def _send(self, req: dict[str, Any], *, expect_type: str) -> dict[str, Any]:
        assert self._proc is not None
        if self._proc.poll() is not None:
            stderr_text = ""
            if self._proc.stderr is not None:
                try:
                    stderr_text = self._proc.stderr.read()
                except Exception:
                    stderr_text = ""
            raise RuntimeError(
                f"Legacy VeoRL bridge already exited with code {self._proc.returncode}."
                + (f"\n[stderr]\n{stderr_text}" if stderr_text else "")
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        return self._read_message(expect_type=expect_type)

    def reset(self) -> dict[str, Any]:
        msg = self._send({"cmd": "reset"}, expect_type="reset")
        obs = _decode_obs(msg["obs"])
        self._last_obs = obs
        self._last_info = None
        return obs

    def step(self, action: np.ndarray):
        msg = self._send(
            {"cmd": "step", "action": np.asarray(action, dtype=np.float32).tolist()},
            expect_type="step",
        )
        obs = _decode_obs(msg["obs"])
        reward = float(msg["reward"])
        done = bool(msg["done"])
        info = msg.get("info", {})
        self._last_obs = obs
        self._last_info = info
        return obs, reward, done, info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only render(mode='rgb_array') is supported for VeoRLBridgeEnv.")
        if self._last_obs is None:
            raise RuntimeError("render() called before reset().")
        if "image_ori" in self._last_obs:
            return np.asarray(self._last_obs["image_ori"])
        return np.asarray(self._last_obs["image"])

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._send({"cmd": "close"}, expect_type="close")
        except Exception:
            pass
        finally:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2.0)
            except Exception:
                pass
            self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
