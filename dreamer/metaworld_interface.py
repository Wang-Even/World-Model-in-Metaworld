from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Tuple, Union

import numpy as np


class MetaWorldLikeEnv(Protocol):
    def reset(self) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: ...

    def render(self, mode: str = "human") -> np.ndarray: ...


PolicyInput = Union[np.ndarray, "jax.Array"]  # type: ignore[name-defined]
PolicyOutput = Union[np.ndarray, "jax.Array", int, float]  # type: ignore[name-defined]


class PolicyFn(Protocol):
    def __call__(self, obs: PolicyInput) -> PolicyOutput: ...


ActionMapping = Union[
    np.ndarray,
    Callable[[PolicyOutput], np.ndarray],
]


@dataclass
class EpisodeResult:
    frames: List[np.ndarray]
    rewards: List[float]
    actions: List[np.ndarray]

    @property
    def total_reward(self) -> float:
        return float(np.sum(self.rewards)) if self.rewards else 0.0


def rollout_metaworld_episode(
    env: MetaWorldLikeEnv,
    policy_fn: PolicyFn,
    *,
    horizon: int = 200,
    use_pixels: bool = True,
    action_mapping: Optional[ActionMapping] = None,
    render_mode: str = "rgb_array",
    initial_obs=None,
    reset_env: bool = True,
) -> EpisodeResult:
    """
    在 Meta-World 环境上运行一个 episode，用给定策略进行控制并返回奖励。

    参数:
        env: 已经构建好的 Meta-World 环境实例 (例如 ML1 或单个任务 env)。
        policy_fn: 策略函数，输入为观测 (像素或低维向量)，输出为:
            - 若提供 action_mapping: 离散动作 id 或任意可映射的值；
            - 若未提供 action_mapping: 直接为连续动作向量。
        horizon: 最大步数。
        use_pixels: 为 True 时，将 env.render(...) 的图像作为策略输入；
                    为 False 时，将 env.reset/step 返回的 obs 作为策略输入。
        action_mapping:
            - 若为 np.ndarray，形状为 (A, act_dim)，则 policy_fn 的输出视为整数索引；
            - 若为 Callable，则使用 action_mapping(policy_output) 生成连续动作；
            - 若为 None，则假设 policy_fn 直接输出连续动作。
        render_mode: 传给 env.render 的模式，一般为 "rgb_array"。
    返回:
        EpisodeResult，包含该 episode 的帧序列、动作、逐步奖励和总奖励。
    """
    def _native_camera_render(e, camera_name: str) -> np.ndarray | None:
        try:
            import mujoco
        except Exception:
            return None

        model = getattr(e, "model", None)
        data = getattr(e, "data", None)
        if model is None or data is None:
            return None

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Unknown MuJoCo camera_name={camera_name!r}")

        renderer = getattr(e, "_codex_native_renderer", None)
        width = int(getattr(e, "_codex_render_width", 256))
        height = int(getattr(e, "_codex_render_height", 256))

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
            e._codex_native_renderer = renderer

        renderer.update_scene(data, camera=camera_name)
        return np.asarray(renderer.render())

    def _call_render(e) -> np.ndarray:
        try:
            return e.render()
        except (TypeError, AttributeError):
            try:
                return e.render(mode=render_mode)
            except (TypeError, AttributeError):
                if hasattr(e, "mujoco_renderer"):
                    return e.mujoco_renderer.render(render_mode)
                raise

    def _safe_render(e) -> np.ndarray:
        """
        兼容不同版本 Gym/Mujoco 的渲染接口：

        优先策略：
          1) 如果显式指定了固定 camera_name，则优先走 MuJoCo 原生 Renderer(camera=...)；
          2) 否则走环境默认 render() 路径；
          3) 若签名不匹配，再尝试 env.render(mode=render_mode)；
          4) 最后退化到 mujoco_renderer.render(render_mode)（若存在）。
        """
        camera_name = getattr(e, "_codex_camera_name", None)
        if camera_name:
            frame = _native_camera_render(e, camera_name)
            if frame is not None:
                return frame

        # 尽量让基类 MujocoEnv.render 使用正确的 render_mode
        try:
            if hasattr(e, "render_mode") and getattr(e, "render_mode", None) is None:
                setattr(e, "render_mode", render_mode)
        except Exception:
            pass

        return _call_render(e)

    if reset_env:
        obs = env.reset()
    else:
        if initial_obs is None:
            raise ValueError("reset_env=False 时必须提供 initial_obs。")
        obs = initial_obs
    if use_pixels:
        if isinstance(obs, dict) and "image" in obs:
            obs_for_policy = obs["image"]
        else:
            frame = _safe_render(env)
            obs_for_policy = frame
    else:
        obs_for_policy = obs

    frames: List[np.ndarray] = []
    rewards: List[float] = []
    actions: List[np.ndarray] = []

    for _ in range(horizon):
        policy_out = policy_fn(obs_for_policy)

        if isinstance(action_mapping, np.ndarray):
            idx = int(np.asarray(policy_out))
            idx_clipped = int(np.clip(idx, 0, action_mapping.shape[0] - 1))
            action = np.asarray(action_mapping[idx_clipped], dtype=np.float32)
        elif callable(action_mapping):
            action = np.asarray(action_mapping(policy_out), dtype=np.float32)
        else:
            action = np.asarray(policy_out, dtype=np.float32)

        # 兼容 Gym / Gymnasium 两种 step API：
        # - 旧版: obs, reward, done, info
        # - 新版: obs, reward, terminated, truncated, info
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs_next, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs_next, reward, done, info = step_out
        else:
            raise ValueError(f"Unsupported env.step return format: {type(step_out)}, len={len(step_out) if isinstance(step_out, tuple) else 'N/A'}")

        # Meta-World 通常在 info 中提供 success / is_success 标志，
        # 即使 done=False 也可能已经成功完成任务，这里也视为 episode 结束。
        success = False
        if isinstance(info, dict):
            if "success" in info:
                success = bool(info["success"])
            elif "is_success" in info:
                success = bool(info["is_success"])

        if use_pixels:
            if isinstance(obs_next, dict) and "image" in obs_next:
                frames.append(np.asarray(obs_next.get("image_ori", obs_next["image"])))
                obs_for_policy = obs_next["image"]
            else:
                frame = _safe_render(env)
                frames.append(np.asarray(frame))
                obs_for_policy = frame
        else:
            frames.append(np.asarray(obs_next))
            obs_for_policy = obs_next

        rewards.append(float(reward))
        actions.append(action)

        # 1) 环境触发 done（终止或截断）
        # 2) 或 info 标记 success / is_success
        # 满足任一条件都提前结束；否则最多执行 horizon 步。
        if done or success:
            break

    return EpisodeResult(frames=frames, rewards=rewards, actions=actions)


def discrete_action_table(actions: Iterable[Iterable[float]]) -> np.ndarray:
    """
    将一个离散动作列表转换为 (A, act_dim) 的查表数组，方便与 rollout_metaworld_episode 搭配使用。
    """
    return np.asarray(list(actions), dtype=np.float32)
