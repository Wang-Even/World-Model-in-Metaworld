from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.veorl_metaworld import VeoRLMetaWorld


def _prepare_frame(frame: np.ndarray, H: int, W: int, C: int) -> np.ndarray:
    f = np.asarray(frame)
    if f.ndim != 3 or f.shape[2] != C:
        raise ValueError(f"期望帧 shape=(H,W,{C}), 实际={f.shape}")
    f = f.astype(np.float32)
    if f.max() > 1.0:
        f = f / 255.0
    return f


def make_metaworld_iterator(
    env_name: str,
    batch_size: int,
    time_steps: int,
    H: int,
    W: int,
    C: int,
    action_dim: int = 4,
    seed: int = 0,
    camera_name: str | None = "corner2",
    render_width: int | None = None,
    render_height: int | None = None,
):
    """
    Generate Meta-World batches using the same collection logic as VeoRL wrappers.py:
      - goal-observable env
      - _freeze_rand_vec = False
      - sim.render(..., camera_name=...)
      - cv2.INTER_AREA resize to 64x64 inside the wrapper
    """
    rng_np = np.random.RandomState(seed)
    native_render_width = int(render_width) if render_width is not None else max(int(W), 128)
    native_render_height = int(render_height) if render_height is not None else max(int(H), 128)
    null_action = action_dim

    env_probe = VeoRLMetaWorld(
        env_name,
        seed=seed,
        action_repeat=1,
        size=(native_render_width, native_render_height),
        camera=camera_name,
    )
    act_dim_cont = int(env_probe.action_space.shape[0])
    low = np.asarray(env_probe.action_space.low, dtype=np.float32)
    high = np.asarray(env_probe.action_space.high, dtype=np.float32)
    env_probe.close()

    step = 0.1
    actions_cont = []
    for i in range(act_dim_cont):
        v_pos = np.zeros(act_dim_cont, dtype=np.float32)
        v_pos[i] = step
        actions_cont.append(np.clip(v_pos, low, high))
        v_neg = np.zeros(act_dim_cont, dtype=np.float32)
        v_neg[i] = -step
        actions_cont.append(np.clip(v_neg, low, high))
    if len(actions_cont) < action_dim:
        actions_cont = actions_cont * ((action_dim + len(actions_cont) - 1) // len(actions_cont))
    action_table = np.stack(actions_cont[:action_dim], axis=0)

    def _next_batch(rng: jax.Array):
        del rng
        videos = np.zeros((batch_size, time_steps, H, W, C), dtype=np.float32)
        actions = np.full((batch_size, time_steps), null_action, dtype=np.int32)
        rewards = np.full((batch_size, time_steps), np.nan, dtype=np.float32)

        for b in range(batch_size):
            env = VeoRLMetaWorld(
                env_name,
                seed=int(rng_np.randint(0, 2**31 - 1)),
                action_repeat=1,
                size=(native_render_width, native_render_height),
                camera=camera_name,
            )
            obs = env.reset()
            videos[b, 0] = _prepare_frame(obs["image"], H, W, C)

            for t in range(1, time_steps):
                a_idx = int(rng_np.randint(0, action_dim))
                obs, rew, done, info = env.step(action_table[a_idx])
                videos[b, t] = _prepare_frame(obs["image"], H, W, C)
                actions[b, t] = a_idx
                rewards[b, t] = float(rew)
                success = bool(info.get("success", info.get("is_success", False))) if isinstance(info, dict) else False
                if done or success:
                    for tt in range(t + 1, time_steps):
                        videos[b, tt] = videos[b, t]
                        actions[b, tt] = actions[b, t]
                        rewards[b, tt] = 0.0
                    break
            env.close()

        return (
            jax.random.PRNGKey(int(rng_np.randint(0, 2**31 - 1))),
            (jnp.asarray(videos), jnp.asarray(actions), jnp.asarray(rewards)),
        )

    return _next_batch
