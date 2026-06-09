from __future__ import annotations


import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Optional, Tuple
import random

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import imageio.v2 as imageio
from PIL import Image

from dreamer.metaworld_interface import (
    rollout_metaworld_episode,
)
from dreamer.veorl_bridge import VeoRLBridgeEnv
from dreamer.veorl_metaworld import VeoRLMetaWorld
from dreamer.data import make_iterator
from dreamer.utils import make_state, make_manager, try_restore
from scripts.train_policy import (
    RLConfig,
    initialize_models,
    encode_frames_to_spatial,
    compute_hidden_from_context,
)


def _maybe_adjust_camera_pose(env, camera_name: Optional[str]) -> None:
    if camera_name != "corner2":
        return
    try:
        import mujoco
    except Exception:
        return

    model = getattr(env, "model", None)
    if model is None:
        return
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Unknown MuJoCo camera_name={camera_name!r}")
    model.cam_pos[cam_id][:] = np.asarray([0.75, 0.075, 0.7], dtype=model.cam_pos.dtype)


def make_metaworld_env(
    task_name: str,
    camera_name: Optional[str] = None,
    camera_pos: tuple[float, float, float] | None = None,
    render_width: int = 128,
    render_height: int = 128,
    *,
    backend: str = "local",
    seed: int = 0,
    bridge_python: str = ".conda-veorl/bin/python",
    bridge_server: str = "scripts/veorl_metaworld_server.py",
):
    if backend == "veorl_bridge":
        return VeoRLBridgeEnv(
            task_name,
            camera_name=camera_name or "corner2",
            camera_pos=camera_pos,
            render_width=int(render_width),
            render_height=int(render_height),
            seed=int(seed),
            python_bin=bridge_python,
            server_script=bridge_server,
        )
    return VeoRLMetaWorld(
        task_name,
        seed=seed,
        action_repeat=1,
        size=(int(render_width), int(render_height)),
        camera=camera_name,
        camera_pos=camera_pos,
    )

def load_dreamer_state_from_ckpt(rl_ckpt_dir: str, context_len_override: int | None) -> Tuple[RLConfig, "TrainState"]:
    m_meta = make_manager(rl_ckpt_dir, item_names=("meta",))
    latest = m_meta.latest_step()
    if latest is None:
        raise FileNotFoundError(f"在 {rl_ckpt_dir} 中未找到 RL checkpoint meta")
    restored_meta = m_meta.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    cfg_dict = meta.get("cfg")
    if cfg_dict is None:
        raise ValueError(f"RL meta 中未找到 cfg 字段: {meta.keys()}")
    if context_len_override is not None:
        cfg_dict = {**cfg_dict, "context_length": int(context_len_override)}
    cfg = RLConfig(**cfg_dict)

    next_batch = make_iterator(
        batch_size=cfg.B,
        time_steps=cfg.T,
        height=cfg.H,
        width=cfg.W,
        channels=cfg.C,
        pixels_per_step=cfg.pixels_per_step,
        size_min=cfg.size_min,
        size_max=cfg.size_max,
        hold_min=cfg.hold_min,
        hold_max=cfg.hold_max,
        fg_min_color=0 if cfg.diversify_data else 128,
        fg_max_color=255 if cfg.diversify_data else 128,
        bg_min_color=0 if cfg.diversify_data else 255,
        bg_max_color=255 if cfg.diversify_data else 255,
    )
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, rewards_init) = next_batch(init_rng)
    del rewards_init

    train_state = initialize_models(cfg, frames_init, actions_init)

    m_state = make_manager(
        rl_ckpt_dir,
        max_to_keep=cfg.ckpt_max_to_keep,
        save_interval_steps=cfg.ckpt_save_every,
    )
    rng = jax.random.PRNGKey(0)
    state_example = make_state(train_state.params, train_state.opt_state, rng, step=0)
    restored = try_restore(m_state, state_example, meta_example={})
    if restored is None:
        raise FileNotFoundError(f"在 {rl_ckpt_dir} 中未找到 RL state checkpoint")
    latest_step, r = restored
    train_state.params = r.state["params"]
    train_state.opt_state = r.state["opt_state"]
    rng = r.state["rng"]
    train_state.pi_vars = {
        **train_state.pi_vars,
        "params": train_state.params["pi"],
    }
    train_state.val_vars = {
        **train_state.val_vars,
        "params": train_state.params["val"],
    }
    print(f"[dreamer] Restored RL policy/value from {rl_ckpt_dir} (step {latest_step})")
    return cfg, train_state


def _make_shape_init_batch(cfg: RLConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    next_batch = make_iterator(
        batch_size=cfg.B,
        time_steps=cfg.T,
        height=cfg.H,
        width=cfg.W,
        channels=cfg.C,
        pixels_per_step=cfg.pixels_per_step,
        size_min=cfg.size_min,
        size_max=cfg.size_max,
        hold_min=cfg.hold_min,
        hold_max=cfg.hold_max,
        fg_min_color=0 if cfg.diversify_data else 128,
        fg_max_color=255 if cfg.diversify_data else 128,
        bg_min_color=0 if cfg.diversify_data else 255,
        bg_max_color=255 if cfg.diversify_data else 255,
    )
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, rewards_init) = next_batch(init_rng)
    del rewards_init
    return frames_init, actions_init


def _rl_cfg_from_bc_meta(
    bc_rew_ckpt_dir: str,
    meta: dict,
    context_len_override: int | None,
) -> RLConfig:
    bc_cfg = meta.get("cfg") or {}
    rl_field_names = {f.name for f in fields(RLConfig)}
    cfg_kwargs = {
        name: bc_cfg[name]
        for name in rl_field_names
        if name in bc_cfg and name != "bc_rew_ckpt"
    }
    cfg_kwargs["run_name"] = bc_cfg.get("run_name") or Path(bc_rew_ckpt_dir).resolve().parent.name
    cfg_kwargs["bc_rew_ckpt"] = bc_rew_ckpt_dir
    if context_len_override is not None:
        cfg_kwargs["context_length"] = int(context_len_override)
    return RLConfig(**cfg_kwargs)


def load_dreamer_state_from_bc_ckpt(
    bc_rew_ckpt_dir: str,
    context_len_override: int | None,
) -> Tuple[RLConfig, "TrainState"]:
    m_meta = make_manager(bc_rew_ckpt_dir, item_names=("meta",))
    latest = m_meta.latest_step()
    if latest is None:
        raise FileNotFoundError(f"在 {bc_rew_ckpt_dir} 中未找到 BC/reward checkpoint meta")
    restored_meta = m_meta.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    cfg = _rl_cfg_from_bc_meta(bc_rew_ckpt_dir, meta, context_len_override)
    # Evaluation only needs shape initialization, not the training batch size.
    cfg = replace(cfg, B=1, T=max(int(cfg.context_length), 1))

    frames_init, actions_init = _make_shape_init_batch(cfg)
    train_state = initialize_models(cfg, frames_init, actions_init)
    print(f"[dreamer] Restored encoder/dynamics/task/bc head from {bc_rew_ckpt_dir} (step {latest})")
    return cfg, train_state


class DreamerPolicy:
    def __init__(
        self,
        cfg: RLConfig,
        train_state,
        context_len: int | None = None,
        rotate_180: bool = False,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        use_bc_head: bool = False,
    ):
        self.cfg = cfg
        self.state = train_state
        self.context_len = int(context_len) if context_len is not None else int(cfg.context_length)
        self.rotate_180 = bool(rotate_180)
        self.use_bc_head = bool(use_bc_head)
        self.n_spatial = cfg.enc_n_latents // cfg.packing_factor
        self.step_idx_scalar = int(np.log2(cfg.k_max))
        self.action_low = np.asarray(
            action_low if action_low is not None else [-1.0] * cfg.action_dim,
            dtype=np.float32,
        )
        self.action_high = np.asarray(
            action_high if action_high is not None else [1.0] * cfg.action_dim,
            dtype=np.float32,
        )
        self.task_ids = jnp.zeros((1,), dtype=jnp.int32)
        self.reset()

    def reset(self) -> None:
        self.z_ctx = None
        self.actions_ctx = None
        self.last_action = None
        self.prepared_frames: list[np.ndarray] = []

    def _prepare_frame(self, frame: np.ndarray) -> jnp.ndarray:
        f = np.asarray(frame)
        if f.ndim != 3 or f.shape[2] != self.cfg.C:
            raise ValueError(f"期望像素观测为 (H,W,{self.cfg.C}), 实际得到 {f.shape}")
        if self.rotate_180:
            # 与离线数据视角对齐：先做 180° 旋转，再进入后续裁剪/归一化流程。
            f = np.rot90(f, 2)
        H, W = self.cfg.H, self.cfg.W
        # Match VeoRL data collection: high-res render, then INTER_AREA downsample to 64x64.
        if f.shape[0] != H or f.shape[1] != W:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        f = f.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        self.prepared_frames.append(np.asarray(np.clip(f * 255.0, 0, 255), dtype=np.uint8))
        return jnp.asarray(f[None, None, ...])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        frames_bt = self._prepare_frame(obs)
        z_new = encode_frames_to_spatial(
            frames_bt,
            encoder=self.state.encoder,
            enc_vars=self.state.enc_vars,
            mae_eval_key=self.state.mae_eval_key,
            patch=self.cfg.patch,
            n_spatial=self.n_spatial,
            packing_factor=self.cfg.packing_factor,
        )
        if self.z_ctx is None:
            self.z_ctx = z_new
            self.actions_ctx = jnp.zeros(
                (1, 1, self.cfg.action_dim),
                dtype=jnp.float32,
            )
        else:
            self.z_ctx = jnp.concatenate([self.z_ctx, z_new], axis=1)
            self.z_ctx = self.z_ctx[:, -self.context_len :, :, :]
            if self.last_action is None:
                new_a = jnp.zeros(
                    (1, 1, self.cfg.action_dim),
                    dtype=jnp.float32,
                )
            else:
                new_a = jnp.asarray(self.last_action, dtype=jnp.float32)[None, None, :]
            self.actions_ctx = jnp.concatenate([self.actions_ctx, new_a], axis=1)
            self.actions_ctx = self.actions_ctx[:, -self.context_len :, :]

        h_t = compute_hidden_from_context(
            self.z_ctx,
            self.actions_ctx,
            dynamics=self.state.dynamics,
            task_embedder=self.state.task_embedder,
            dyn_vars=self.state.dyn_vars,
            task_vars=self.state.task_vars,
            task_ids=self.task_ids,
            step_idx_scalar=self.step_idx_scalar,
            k_max=self.cfg.k_max,
        )
        h_for_policy = h_t[:, None, :]  # (B, 1, d_model)
        head = self.state.policy_head_bc if self.use_bc_head else self.state.policy_head
        vars_ = self.state.pi_bc_vars if self.use_bc_head else self.state.pi_vars
        pi_pred = head.apply(
            vars_,
            h_for_policy,
            deterministic=True,
        )  # (B, 1, L, A)
        action = np.asarray(pi_pred[0, 0, 0, :], dtype=np.float32)
        action = np.clip(action, self.action_low, self.action_high)
        self.last_action = action
        return action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simple Dreamer-style interface on Meta-World."
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Meta-World 任务名，例如 drawer-open-v3, pick-place-v2 等。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="占位参数，用于兼容类似 `--config dynamics/soar-small` 的调用；当前脚本中未实际使用。",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=8,
        help="上下文长度（预留给后续接 Dreamer 世界模型用）。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="运行多少条 episode。",
    )
    parser.add_argument(
        "--rollout_len",
        type=int,
        default=64,
        help="每条 episode 的最大步数。",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子（当前只用于打印，未参与更新）。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="学习率（当前只用于打印，未参与更新）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子。",
    )
    parser.add_argument(
        "--rotate_180",
        action="store_true",
        help="可选: 将输入给策略的像素观测旋转 180°。VeoRL 原始采样逻辑默认不旋转。",
    )
    parser.add_argument(
        "--no_rotate_180",
        action="store_false",
        dest="rotate_180",
        help="关闭 180° 旋转。",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="corner",
        help="Meta-World 相机名，按 VeoRL wrapper 透传给 sim.render(camera_name=...)。",
    )
    parser.add_argument(
        "--camera_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="可选: 覆盖 MuJoCo 相机位置，例如 --camera_pos 0.91 0.03 0.73。",
    )
    parser.add_argument(
        "--render_width",
        type=int,
        default=128,
        help="MuJoCo 原生固定相机渲染宽度。",
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=128,
        help="MuJoCo 原生固定相机渲染高度。",
    )
    parser.add_argument(
        "--use_bc_head",
        action="store_true",
        help="使用 BC 头控制，而不是 RL policy 头。",
    )
    parser.add_argument(
        "--rl_ckpt",
        type=str,
        default="",
        help="Dreamer RL checkpoint 目录（train_policy.py 训练产生的 checkpoints 路径）。",
    )
    parser.add_argument(
        "--bc_rew_ckpt",
        type=str,
        default="",
        help="BC/reward checkpoint 目录（train_bc_rew_heads.py 训练产生的 checkpoints 路径）。",
    )
    parser.add_argument(
        "--save_preview_episodes",
        type=int,
        default=1,
        help="保存前多少个 episode 的抽帧预览图。",
    )
    parser.add_argument(
        "--save_preview_frames",
        type=int,
        default=8,
        help="每个预览 episode 保存多少帧。",
    )
    parser.add_argument(
        "--env_backend",
        type=str,
        default="local",
        choices=["local", "veorl_bridge"],
        help="推理环境后端：local 使用当前 Python 环境；veorl_bridge 通过旧版 VeoRL 环境子进程采样。",
    )
    parser.add_argument(
        "--bridge_python",
        type=str,
        default=".conda-veorl/bin/python",
        help="veorl_bridge 后端使用的 Python 可执行文件。",
    )
    parser.add_argument(
        "--bridge_server",
        type=str,
        default="scripts/veorl_metaworld_server.py",
        help="veorl_bridge 后端使用的 server 脚本路径。",
    )

    parser.set_defaults(rotate_180=False)
    args = parser.parse_args()

    # 统一设置 numpy 和 Python random 的种子（便于复现）
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"[info] env={args.env}")
    print(f"[info] config={args.config} (占位，不影响当前脚本)")
    print(f"[info] context_len={args.context_len}, rollout_len={args.rollout_len}")
    print(f"[info] episodes={args.episodes}, gamma={args.gamma}, lr={args.lr}")
    print(f"[info] camera_name={args.camera_name}")
    print(f"[info] camera_pos={tuple(args.camera_pos) if args.camera_pos is not None else None}")
    print(f"[info] render_size={args.render_width}x{args.render_height}")
    print(f"[info] env_backend={args.env_backend}")
    print(f"[info] control_head={'bc' if args.use_bc_head else 'rl'}")
    if args.use_bc_head and not args.rl_ckpt:
        if not args.bc_rew_ckpt:
            raise ValueError("使用 --use_bc_head 且不提供 --rl_ckpt 时，必须提供 --bc_rew_ckpt。")
        cfg, train_state = load_dreamer_state_from_bc_ckpt(args.bc_rew_ckpt, args.context_len)
    else:
        if not args.rl_ckpt:
            raise ValueError("使用 RL 头评估时必须提供 --rl_ckpt。")
        cfg, train_state = load_dreamer_state_from_ckpt(args.rl_ckpt, args.context_len)

    camera_pos = tuple(args.camera_pos) if args.camera_pos is not None else None

    env_probe = make_metaworld_env(
        args.env,
        camera_name=args.camera_name,
        camera_pos=camera_pos,
        render_width=args.render_width,
        render_height=args.render_height,
        backend=args.env_backend,
        seed=args.seed,
        bridge_python=args.bridge_python,
        bridge_server=args.bridge_server,
    )
    print(f"[info] env_action_dim={env_probe.action_space.shape[0]}, dreamer_action_dim={cfg.action_dim}")

    dreamer_policy = DreamerPolicy(
        cfg,
        train_state,
        context_len=args.context_len,
        rotate_180=args.rotate_180,
        action_low=np.asarray(env_probe.action_space.low, dtype=np.float32),
        action_high=np.asarray(env_probe.action_space.high, dtype=np.float32),
        use_bc_head=args.use_bc_head,
    )
    try:
        env_probe.close()
    except Exception:
        pass

    returns = []
    out_dir = Path("logs") / f"metaworld_{args.env}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可视化时我们直接使用环境原始渲染分辨率，不做裁剪，
    # 这样视野范围最大，尽量把机械臂和按钮都包含在画面中。

    for ep in range(args.episodes):
        env = make_metaworld_env(
            args.env,
            camera_name=args.camera_name,
            camera_pos=camera_pos,
            render_width=args.render_width,
            render_height=args.render_height,
            backend=args.env_backend,
            seed=args.seed + ep,
            bridge_python=args.bridge_python,
            bridge_server=args.bridge_server,
        )
        try:
            initial_obs = env.reset()
            first_raw = None
            if isinstance(initial_obs, dict):
                first_raw = initial_obs.get("image_ori")
            first_policy_input = None
            if isinstance(initial_obs, dict):
                first_policy_input = initial_obs.get("image")

            dreamer_policy.reset()
            ep_result = rollout_metaworld_episode(
                env,
                dreamer_policy,
                horizon=args.rollout_len,
                use_pixels=True,
                initial_obs=initial_obs,
                reset_env=False,
            )

            G = float(np.sum(ep_result.rewards))
            returns.append(G)
            print(
                f"[episode {ep:04d}] steps={len(ep_result.rewards):03d} "
                f"return={G:.3f}"
            )

            # 保存环境原始渲染帧，以及策略实际输入的预处理帧。
            if ep < args.save_preview_episodes:
                ep_dir = out_dir / f"episode_{ep:04d}_preview"
                raw_dir = ep_dir / "raw_frames"
                prepared_dir = ep_dir / "policy_input_frames"
                raw_dir.mkdir(parents=True, exist_ok=True)
                prepared_dir.mkdir(parents=True, exist_ok=True)

                if ep_result.frames:
                    frames_np = np.asarray(ep_result.frames)  # (T, H, W, C)
                    num_frames_to_save = min(args.save_preview_frames, frames_np.shape[0])
                    for t in range(num_frames_to_save):
                        img = frames_np[t]  # 使用完整视野
                        img_path = raw_dir / f"frame_{t:03d}.png"
                        try:
                            imageio.imwrite(img_path, img)
                        except Exception as e:
                            print(f"[episode {ep:04d}] failed to save raw frame {t}: {e}")

                prepared_np = np.asarray(dreamer_policy.prepared_frames)
                if prepared_np.size:
                    num_prepared_to_save = min(args.save_preview_frames, prepared_np.shape[0])
                    for t in range(num_prepared_to_save):
                        img = prepared_np[t]
                        img_path = prepared_dir / f"frame_{t:03d}.png"
                        try:
                            imageio.imwrite(img_path, img)
                        except Exception as e:
                            print(f"[episode {ep:04d}] failed to save policy-input frame {t}: {e}")

                if prepared_np.size:
                    try:
                        imageio.imwrite(ep_dir / "episode_first_policy_input.png", prepared_np[0])
                    except Exception as e:
                        print(f"[episode {ep:04d}] failed to save first policy input: {e}")

                if first_raw is not None:
                    try:
                        imageio.imwrite(ep_dir / "episode_first_raw.png", np.asarray(first_raw))
                    except Exception as e:
                        print(f"[episode {ep:04d}] failed to save first raw frame: {e}")
                if first_policy_input is not None:
                    try:
                        imageio.imwrite(ep_dir / "episode_first_env_image.png", np.asarray(first_policy_input))
                    except Exception as e:
                        print(f"[episode {ep:04d}] failed to save first env image: {e}")

                print(
                    f"[episode {ep:04d}] saved preview frames to {ep_dir}"
                )
        finally:
            try:
                env.close()
            except Exception:
                pass

    if returns:
        returns_arr = np.asarray(returns, dtype=np.float32)
        print(
            f"[summary] episodes={len(returns_arr)}, "
            f"mean_return={returns_arr.mean():.3f}, "
            f"std={returns_arr.std():.3f}, "
            f"min={returns_arr.min():.3f}, "
            f"max={returns_arr.max():.3f}"
        )


if __name__ == "__main__":
    main()
    
