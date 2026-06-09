from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def _normalize_task_name(name: str) -> str:
    base = name.replace("_", "-")
    for suffix in ("-v3", "-v2", "-goal-observable", "-goal-hidden"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


class VeoRLMetaWorld:
    """
    Meta-World wrapper aligned to VeoRL's wrappers.py:
      - ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
      - _freeze_rand_vec = False
      - sim.render(..., mode="offscreen", camera_name=...)
      - cv2.INTER_AREA resize to 64x64 for the returned `image`
      - keep raw render in `image_ori`
      - step() always returns done=False; outer time-limit / success handles episode stop
    """

    def __init__(
        self,
        name: str,
        seed: Optional[int] = None,
        action_repeat: int = 1,
        size: tuple[int, int] = (64, 64),
        camera: Optional[str] = None,
        camera_pos: Optional[tuple[float, float, float]] = None,
        camera_fovy: Optional[float] = None,
    ):
        import metaworld
        try:
            from metaworld import env_dict
        except Exception:
            from metaworld.envs.mujoco import env_dict

        del metaworld
        os.environ.setdefault("MUJOCO_GL", "egl")

        base = _normalize_task_name(name)
        candidate_dicts = []
        if hasattr(env_dict, "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE"):
            candidate_dicts.append(("ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE", env_dict.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, f"{base}-v2-goal-observable"))
        if hasattr(env_dict, "ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE"):
            candidate_dicts.append(("ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE", env_dict.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE, f"{base}-v3-goal-observable"))

        env_cls = None
        chosen = None
        for dict_name, envs, task in candidate_dicts:
            if task in envs:
                env_cls = envs[task]
                chosen = (dict_name, task)
                break
        if env_cls is None:
            raise KeyError(
                f"Task {name!r} not found in available goal-observable env dicts. "
                f"Tried {[task for _, _, task in candidate_dicts]}."
            )
        try:
            self._env = env_cls(seed=seed)
        except TypeError:
            self._env = env_cls()
        self._env._freeze_rand_vec = False
        self._size = tuple(size)
        self._action_repeat = int(action_repeat)
        self._camera = camera
        self._camera_pos = tuple(camera_pos) if camera_pos is not None else None
        self._camera_fovy = float(camera_fovy) if camera_fovy is not None else None
        self._last_obs = None
        self._chosen_env_key = chosen

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    def close(self):
        renderer = getattr(self, "_legacy_native_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        return self._env.close()

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._last_obs is None:
            return self._render_image()
        return self._last_obs["image_ori"]

    def reset(self):
        self._apply_camera_overrides()
        state = self._env.reset()
        raw = self._render_image()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._resize_to_64(raw),
            "image_ori": raw,
            "state": state,
            "success": False,
        }
        self._last_obs = obs
        return {"image": obs["image"], "image_ori": obs["image_ori"], "state": obs["state"]}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert np.isfinite(action).all(), action
        reward = 0.0
        success = 0.0
        info = {}
        state = None
        for _ in range(self._action_repeat):
            step_out = self._env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                state, rew, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            else:
                state, rew, done, info = step_out
            reward += rew or 0.0
            success += float(info.get("success", 0.0))
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        raw = self._render_image()
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
            "image": self._resize_to_64(raw),
            "image_ori": raw,
            "state": state,
            "success": success,
        }
        self._last_obs = obs
        return {"image": obs["image"], "image_ori": obs["image_ori"], "state": obs["state"]}, reward, False, info

    def _render_image(self) -> np.ndarray:
        if hasattr(self._env, "sim") and getattr(self._env, "sim") is not None:
            return self._render_legacy_sim_with_native_mujoco()

        import mujoco

        model = getattr(self._env, "model", None)
        data = getattr(self._env, "data", None)
        if model is None or data is None:
            raise AttributeError("Meta-World env has neither sim.render nor model/data for offscreen rendering.")

        renderer = getattr(self, "_native_renderer", None)
        width, height = int(self._size[0]), int(self._size[1])
        needs_new = True
        if renderer is not None:
            try:
                needs_new = (
                    int(getattr(renderer, "width", -1)) != width
                    or int(getattr(renderer, "height", -1)) != height
                )
            except Exception:
                needs_new = True
        if needs_new:
            try:
                if renderer is not None:
                    renderer.close()
            except Exception:
                pass
            renderer = mujoco.Renderer(model, height, width)
            self._native_renderer = renderer

        renderer.update_scene(data, camera=self._camera)
        return np.asarray(renderer.render())

    def _render_legacy_sim_with_native_mujoco(self) -> np.ndarray:
        """
        Legacy Meta-World + mujoco-py transitions, but rendering via the modern
        `mujoco.Renderer`. This preserves the old environment/task definition
        and camera layout while avoiding unstable `sim.render(offscreen)` calls.
        """
        import mujoco

        width, height = int(self._size[0]), int(self._size[1])
        native_model = getattr(self, "_legacy_native_model", None)
        native_data = getattr(self, "_legacy_native_data", None)
        renderer = getattr(self, "_legacy_native_renderer", None)
        native_model_path = getattr(self, "_legacy_native_model_path", None)
        model_path = getattr(self._env, "model_name", None)

        if native_model is None or native_data is None or native_model_path != model_path:
            if model_path is None:
                xml = self._env.sim.model.get_xml()
                native_model = mujoco.MjModel.from_xml_string(xml)
            else:
                native_model = mujoco.MjModel.from_xml_path(model_path)
            native_data = mujoco.MjData(native_model)
            self._legacy_native_model = native_model
            self._legacy_native_data = native_data
            self._legacy_native_model_path = model_path
            renderer = None

        needs_new = True
        if renderer is not None:
            try:
                needs_new = (
                    int(getattr(renderer, "width", -1)) != width
                    or int(getattr(renderer, "height", -1)) != height
                )
            except Exception:
                needs_new = True
        if needs_new:
            try:
                if renderer is not None:
                    renderer.close()
            except Exception:
                pass
            renderer = mujoco.Renderer(native_model, height, width)
            self._legacy_native_renderer = renderer

        # Keep camera pose/fov aligned with the legacy env, including runtime tweaks
        # like VeoRL's `corner2` cam_pos override.
        for attr in ("cam_pos", "cam_quat", "cam_fovy"):
            if hasattr(self._env.model, attr) and hasattr(native_model, attr):
                np.copyto(getattr(native_model, attr), np.asarray(getattr(self._env.model, attr)))

        sim_data = self._env.sim.data
        native_data.qpos[:] = np.asarray(sim_data.qpos).ravel()
        native_data.qvel[:] = np.asarray(sim_data.qvel).ravel()
        if native_model.na:
            native_data.act[:] = np.asarray(sim_data.act).ravel()
        if native_model.nmocap:
            native_data.mocap_pos[:] = np.asarray(sim_data.mocap_pos)
            native_data.mocap_quat[:] = np.asarray(sim_data.mocap_quat)
        if native_model.nuserdata:
            native_data.userdata[:] = np.asarray(sim_data.userdata).ravel()
        native_data.time = float(sim_data.time)
        mujoco.mj_forward(native_model, native_data)

        renderer.update_scene(native_data, camera=self._camera)
        return np.asarray(renderer.render())

    def _apply_camera_overrides(self) -> None:
        if self._camera != "corner2":
            return
        try:
            cam_idx = 2
            if self._camera_pos is not None:
                self._env.model.cam_pos[cam_idx][:] = list(self._camera_pos)
            else:
                self._env.model.cam_pos[cam_idx][:] = [0.75, 0.075, 0.7]
            if self._camera_fovy is not None and hasattr(self._env.model, "cam_fovy"):
                self._env.model.cam_fovy[cam_idx] = float(self._camera_fovy)
        except Exception:
            pass

    @staticmethod
    def _resize_to_64(image: np.ndarray) -> np.ndarray:
        return cv2.resize(np.asarray(image), (64, 64), interpolation=cv2.INTER_AREA)
