from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image, ImageDraw

from dreamer.data import make_iterator
from dreamer.sampler import denoise_single_latent
from dreamer.veorl_bridge import VeoRLBridgeEnv
from dreamer.utils import (
    make_manager,
    temporal_unpatchify,
    unpack_spatial_to_bottleneck,
    with_params,
)
from scripts.train_policy import (
    RLConfig,
    compute_hidden_from_context,
    encode_frames_to_spatial,
    initialize_models,
)
from scripts.train_dynamics import (
    RealismConfig as DynamicsConfig,
    initialize_models_and_tokenizer as initialize_dynamics_models_and_tokenizer,
)
from dreamer.veorl_metaworld import VeoRLMetaWorld


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
    )


def _safe_render(env, render_mode: str = "rgb_array") -> np.ndarray:
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

    camera_name = getattr(env, "_codex_camera_name", None)
    if camera_name:
        frame = _native_camera_render(env, camera_name)
        if frame is not None:
            return frame

    try:
        return env.render()
    except (TypeError, AttributeError):
        return env.render(mode=render_mode)


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
    cfg = replace(cfg, B=1, T=max(int(cfg.context_length), 1))
    frames_init, actions_init = _make_shape_init_batch(cfg)
    train_state = initialize_models(cfg, frames_init, actions_init)
    print(f"[dreamer] Restored encoder/dynamics/task/bc head from {bc_rew_ckpt_dir} (step {latest})")
    return cfg, train_state


def load_dynamics_only_state_from_ckpt(
    dynamics_ckpt_dir: str,
    *,
    context_len: int,
    action_dim: int = 4,
) -> Tuple[dict, "TrainState"]:
    m_meta = make_manager(dynamics_ckpt_dir, item_names=("meta",))
    latest = m_meta.latest_step()
    if latest is None:
        raise FileNotFoundError(f"在 {dynamics_ckpt_dir} 中未找到 dynamics checkpoint meta")
    restored_meta = m_meta.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    cfg_dict = dict(meta.get("cfg") or {})
    dyn_cfg = DynamicsConfig(**cfg_dict)

    B = 1
    T = max(int(context_len), 1)
    H = int(meta["H"])
    W = int(meta["W"])
    C = int(meta["C"])
    frames_init = jnp.zeros((B, T, H, W, C), dtype=jnp.float32)
    actions_init = jnp.zeros((B, T, action_dim), dtype=jnp.float32)

    train_state = initialize_dynamics_models_and_tokenizer(dyn_cfg, frames_init, actions_init)
    state_example = {
        "params": train_state.params,
        "opt_state": train_state.opt_state,
        "rng": jax.random.PRNGKey(0),
        "step": jnp.int32(0),
    }
    m_state = make_manager(dynamics_ckpt_dir, item_names=("state", "meta"))
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)
    restored = m_state.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    train_state.params = restored.state["params"]
    train_state.opt_state = restored.state["opt_state"]
    train_state.dyn_vars = with_params(train_state.dyn_vars, train_state.params)
    eval_cfg = {
        "H": H,
        "W": W,
        "C": C,
        "patch": int(meta["patch"]),
        "k_max": int(meta["k_max"]),
        "packing_factor": int(meta["packing_factor"]),
        "enc_n_latents": int(meta["enc_kwargs"]["n_latents"]),
        "enc_d_bottleneck": int(meta["enc_kwargs"]["d_bottleneck"]),
        "context_length": int(context_len),
        "action_dim": int(action_dim),
    }
    print(f"[dreamer] Restored encoder/decoder/dynamics from {dynamics_ckpt_dir} (step {latest})")
    return eval_cfg, train_state


@dataclass
class StepMetrics:
    mse: float
    mae: float
    rmse: float
    psnr: float


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> StepMetrics:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    psnr = float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
    return StepMetrics(mse=mse, mae=mae, rmse=rmse, psnr=psnr)


def _to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    return np.asarray(np.clip(x * 255.0, 0, 255), dtype=np.uint8)


def _make_panel(curr: np.ndarray, pred: np.ndarray, gt: np.ndarray, metrics: StepMetrics) -> np.ndarray:
    curr_u8 = _to_u8(curr)
    pred_u8 = _to_u8(pred)
    gt_u8 = _to_u8(gt)

    def _with_title(img: np.ndarray, title: str) -> np.ndarray:
        im = Image.fromarray(img)
        canvas = Image.new("RGB", (im.width, im.height + 20), (255, 255, 255))
        canvas.paste(im, (0, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), title, fill=(0, 0, 0))
        return np.asarray(canvas)

    tiles = np.concatenate(
        [
            _with_title(curr_u8, "prev frame"),
            _with_title(pred_u8, f"pred next | PSNR {metrics.psnr:.2f}"),
            _with_title(gt_u8, "real next"),
        ],
        axis=1,
    )
    return tiles


class OnlineWorldModelEvaluator:
    def __init__(
        self,
        cfg,
        train_state,
        *,
        context_len: int,
        rotate_180: bool,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_mode: str,
        seed: int,
    ):
        self.cfg = cfg
        self.state = train_state
        self.context_len = int(context_len)
        self.rotate_180 = bool(rotate_180)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_mode = action_mode
        self.n_spatial = int(cfg["enc_n_latents"] // cfg["packing_factor"]) if isinstance(cfg, dict) else cfg.enc_n_latents // cfg.packing_factor
        self.step_idx_scalar = int(np.log2(cfg["k_max"])) if isinstance(cfg, dict) else int(np.log2(cfg.k_max))
        self.task_ids = jnp.zeros((1,), dtype=jnp.int32)
        self.rng = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self) -> None:
        self.z_ctx = None
        self.actions_ctx = None
        self.last_action = None

    def preprocess_raw_frame(self, frame: np.ndarray) -> tuple[jnp.ndarray, np.ndarray]:
        f = np.asarray(frame)
        C = int(self.cfg["C"]) if isinstance(self.cfg, dict) else self.cfg.C
        H = int(self.cfg["H"]) if isinstance(self.cfg, dict) else self.cfg.H
        W = int(self.cfg["W"]) if isinstance(self.cfg, dict) else self.cfg.W
        if f.ndim != 3 or f.shape[2] != C:
            raise ValueError(f"期望像素观测为 (H,W,{C}), 实际得到 {f.shape}")
        if self.rotate_180:
            f = np.rot90(f, 2)
        if f.shape[0] != H or f.shape[1] != W:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        f = f.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        return jnp.asarray(f[None, None, ...]), np.asarray(np.clip(f * 255.0, 0, 255), dtype=np.uint8)

    def observe_prepared_frame(self, frame_bt: jnp.ndarray) -> None:
        z_new = encode_frames_to_spatial(
            frame_bt,
            encoder=self.state.encoder,
            enc_vars=self.state.enc_vars,
            mae_eval_key=self.state.mae_eval_key,
            patch=int(self.cfg["patch"]) if isinstance(self.cfg, dict) else self.cfg.patch,
            n_spatial=self.n_spatial,
            packing_factor=int(self.cfg["packing_factor"]) if isinstance(self.cfg, dict) else self.cfg.packing_factor,
        )
        if self.z_ctx is None:
            self.z_ctx = z_new
            action_dim = int(self.cfg["action_dim"]) if isinstance(self.cfg, dict) else self.cfg.action_dim
            self.actions_ctx = jnp.zeros((1, 1, action_dim), dtype=jnp.float32)
        else:
            self.z_ctx = jnp.concatenate([self.z_ctx, z_new], axis=1)[:, -self.context_len :, :, :]
            if self.last_action is None:
                action_dim = int(self.cfg["action_dim"]) if isinstance(self.cfg, dict) else self.cfg.action_dim
                new_a = jnp.zeros((1, 1, action_dim), dtype=jnp.float32)
            else:
                new_a = jnp.asarray(self.last_action, dtype=jnp.float32)[None, None, :]
            self.actions_ctx = jnp.concatenate([self.actions_ctx, new_a], axis=1)[:, -self.context_len :, :]

    def choose_action(self) -> np.ndarray:
        action_dim = int(self.cfg["action_dim"]) if isinstance(self.cfg, dict) else self.cfg.action_dim
        if self.action_mode == "zero":
            action = np.zeros((action_dim,), dtype=np.float32)
        elif self.action_mode == "random":
            self.rng, rk = jax.random.split(self.rng)
            action = np.asarray(
                jax.random.uniform(
                    rk,
                    (action_dim,),
                    minval=self.action_low,
                    maxval=self.action_high,
                ),
                dtype=np.float32,
            )
        else:
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
            if not hasattr(self.state, "policy_head_bc"):
                raise ValueError("当前只加载了 dynamics checkpoint，不能使用 --action_mode bc。请改用 random 或 zero。")
            pi_pred = self.state.policy_head_bc.apply(
                self.state.pi_bc_vars,
                h_t[:, None, :],
                deterministic=True,
            )
            action = np.asarray(pi_pred[0, 0, 0, :], dtype=np.float32)

        action = np.clip(action, self.action_low, self.action_high)
        self.last_action = action
        return action

    def predict_next_frame(self, action: np.ndarray) -> np.ndarray:
        self.rng, rk = jax.random.split(self.rng)
        z_t_init = jax.random.normal(
            rk,
            (1, 1, self.n_spatial, self.z_ctx.shape[-1]),
            dtype=self.z_ctx.dtype,
        )
        z_pred, _ = denoise_single_latent(
            dynamics=self.state.dynamics,
            dyn_vars=self.state.dyn_vars,
            actions_ctx=self.actions_ctx,
            action_curr=jnp.asarray(action, dtype=jnp.float32)[None, None, :],
            z_ctx_clean=self.z_ctx,
            z_t_init=z_t_init,
            k_max=int(self.cfg["k_max"]) if isinstance(self.cfg, dict) else self.cfg.k_max,
            d=1.0 / float(int(self.cfg["k_max"]) if isinstance(self.cfg, dict) else self.cfg.k_max),
            start_mode="pure",
            tau0_fixed=0.0,
            rng_key=self.rng,
            clean_target_next=None,
            agent_tokens=None,
            match_ctx_tau=False,
        )
        pred_btLd = unpack_spatial_to_bottleneck(
            z_pred,
            n_spatial=self.n_spatial,
            k=int(self.cfg["packing_factor"]) if isinstance(self.cfg, dict) else self.cfg.packing_factor,
        )
        pred_patches = self.state.decoder.apply(
            self.state.dec_vars,
            pred_btLd,
            deterministic=True,
        )
        pred_frames = temporal_unpatchify(
            pred_patches,
            int(self.cfg["H"]) if isinstance(self.cfg, dict) else self.cfg.H,
            int(self.cfg["W"]) if isinstance(self.cfg, dict) else self.cfg.W,
            int(self.cfg["C"]) if isinstance(self.cfg, dict) else self.cfg.C,
            int(self.cfg["patch"]) if isinstance(self.cfg, dict) else self.cfg.patch,
        )
        pred = np.asarray(pred_frames[0, 0], dtype=np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate online one-step Meta-World world-model prediction with real camera inputs."
    )
    parser.add_argument("--env", type=str, required=True, help="Meta-World 任务名，例如 button-press-v3")
    parser.add_argument("--bc_rew_ckpt", type=str, default="", help="BC/reward checkpoint 目录")
    parser.add_argument("--dynamics_ckpt", type=str, default="", help="Dynamics checkpoint 目录")
    parser.add_argument("--episodes", type=int, default=3, help="评估多少个 episode")
    parser.add_argument("--rollout_len", type=int, default=50, help="每个 episode 最多多少步")
    parser.add_argument("--context_len", type=int, default=8, help="上下文长度")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--camera_name", type=str, default="corner2", help="固定相机名")
    parser.add_argument("--render_width", type=int, default=128, help="原生相机渲染宽度")
    parser.add_argument("--render_height", type=int, default=128, help="原生相机渲染高度")
    parser.add_argument("--rotate_180", action="store_true", help="策略输入旋转 180 度")
    parser.add_argument("--no_rotate_180", action="store_false", dest="rotate_180", help="关闭 180 度旋转")
    parser.add_argument("--action_mode", type=str, default="random", choices=["bc", "random", "zero"], help="环境执行动作来源：bc / random / zero")
    parser.add_argument("--save_preview_steps", type=int, default=16, help="每个 episode 最多保存多少步对比图")
    parser.add_argument("--save_video", action="store_true", help="保存每个 episode 的对比 mp4")
    parser.add_argument("--env_backend", type=str, default="local", choices=["local", "veorl_bridge"], help="推理环境后端：local 或 veorl_bridge")
    parser.add_argument("--bridge_python", type=str, default=".conda-veorl/bin/python", help="veorl_bridge 后端使用的 Python 可执行文件")
    parser.add_argument("--bridge_server", type=str, default="scripts/veorl_metaworld_server.py", help="veorl_bridge 后端使用的 server 脚本路径")
    parser.set_defaults(rotate_180=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if bool(args.bc_rew_ckpt) == bool(args.dynamics_ckpt):
        raise ValueError("必须且只能提供一个: --bc_rew_ckpt 或 --dynamics_ckpt。")
    if args.dynamics_ckpt:
        cfg, train_state = load_dynamics_only_state_from_ckpt(
            args.dynamics_ckpt,
            context_len=args.context_len,
            action_dim=4,
        )
    else:
        cfg, train_state = load_dreamer_state_from_bc_ckpt(args.bc_rew_ckpt, args.context_len)
    env = make_metaworld_env(
        args.env,
        camera_name=args.camera_name,
        render_width=args.render_width,
        render_height=args.render_height,
        backend=args.env_backend,
        seed=args.seed,
        bridge_python=args.bridge_python,
        bridge_server=args.bridge_server,
    )

    evaluator = OnlineWorldModelEvaluator(
        cfg,
        train_state,
        context_len=args.context_len,
        rotate_180=args.rotate_180,
        action_low=np.asarray(env.action_space.low, dtype=np.float32),
        action_high=np.asarray(env.action_space.high, dtype=np.float32),
        action_mode=args.action_mode,
        seed=args.seed,
    )

    print(f"[info] env={args.env}")
    print(f"[info] camera_name={args.camera_name}, render={args.render_width}x{args.render_height}, rotate_180={args.rotate_180}")
    print(f"[info] env_backend={args.env_backend}")
    H = int(cfg["H"]) if isinstance(cfg, dict) else cfg.H
    W = int(cfg["W"]) if isinstance(cfg, dict) else cfg.W
    print(f"[info] wm_input={H}x{W}, action_mode={args.action_mode}, context_len={args.context_len}")

    out_dir = Path("logs") / f"wm_online_eval_{args.env}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[StepMetrics] = []
    all_returns: list[float] = []
    episode_summaries: list[dict] = []

    try:
        for ep in range(args.episodes):
            evaluator.reset()
            obs = env.reset()
            raw_frame = obs["image_ori"] if isinstance(obs, dict) and "image_ori" in obs else _safe_render(env)
            policy_frame = obs["image"] if isinstance(obs, dict) and "image" in obs else raw_frame
            curr_frame_bt, curr_frame_u8 = evaluator.preprocess_raw_frame(policy_frame)

            ep_dir = out_dir / f"episode_{ep:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            panels: list[np.ndarray] = []
            ep_metrics: list[StepMetrics] = []
            ep_rewards: list[float] = []

            for step in range(args.rollout_len):
                evaluator.observe_prepared_frame(curr_frame_bt)
                action = evaluator.choose_action()
                pred_next = evaluator.predict_next_frame(action)

                step_out = env.step(action)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    obs_next, reward, terminated, truncated, info = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    obs_next, reward, done, info = step_out

                raw_next = obs_next["image_ori"] if isinstance(obs_next, dict) and "image_ori" in obs_next else _safe_render(env)
                policy_next = obs_next["image"] if isinstance(obs_next, dict) and "image" in obs_next else raw_next
                next_frame_bt, next_frame_u8 = evaluator.preprocess_raw_frame(policy_next)
                gt_next = np.asarray(next_frame_bt[0, 0], dtype=np.float32)

                metrics = _compute_metrics(pred_next, gt_next)
                ep_metrics.append(metrics)
                ep_rewards.append(float(reward))
                all_metrics.append(metrics)

                print(
                    f"[episode {ep:04d} step {step:03d}] "
                    f"reward={float(reward):.3f} mse={metrics.mse:.6f} "
                    f"mae={metrics.mae:.6f} psnr={metrics.psnr:.2f}dB"
                )

                if step < args.save_preview_steps:
                    panel = _make_panel(
                        curr=curr_frame_bt[0, 0],
                        pred=pred_next,
                        gt=gt_next,
                        metrics=metrics,
                    )
                    panels.append(panel)
                    imageio.imwrite(ep_dir / f"step_{step:03d}.png", panel)

                curr_frame_bt, curr_frame_u8 = next_frame_bt, next_frame_u8

                success = False
                if isinstance(info, dict):
                    success = bool(info.get("success", info.get("is_success", False)))
                if done or success:
                    break

            if args.save_video and panels:
                imageio.mimwrite(ep_dir / "prediction_panels.mp4", panels, fps=4)

            ep_return = float(np.sum(ep_rewards))
            all_returns.append(ep_return)
            ep_summary = {
                "episode": ep,
                "steps": len(ep_metrics),
                "return": ep_return,
                "mean_mse": float(np.mean([m.mse for m in ep_metrics])) if ep_metrics else None,
                "mean_mae": float(np.mean([m.mae for m in ep_metrics])) if ep_metrics else None,
                "mean_psnr": float(np.mean([m.psnr for m in ep_metrics])) if ep_metrics else None,
            }
            episode_summaries.append(ep_summary)
            print(
                f"[episode {ep:04d}] steps={ep_summary['steps']} return={ep_return:.3f} "
                f"mean_mse={ep_summary['mean_mse']:.6f} mean_mae={ep_summary['mean_mae']:.6f} "
                f"mean_psnr={ep_summary['mean_psnr']:.2f}dB"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    summary = {
        "env": args.env,
        "bc_rew_ckpt": args.bc_rew_ckpt,
        "dynamics_ckpt": args.dynamics_ckpt,
        "camera_name": args.camera_name,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "rotate_180": args.rotate_180,
        "wm_input_size": [
            int(cfg["H"]) if isinstance(cfg, dict) else cfg.H,
            int(cfg["W"]) if isinstance(cfg, dict) else cfg.W,
            int(cfg["C"]) if isinstance(cfg, dict) else cfg.C,
        ],
        "episodes": args.episodes,
        "rollout_len": args.rollout_len,
        "action_mode": args.action_mode,
        "mean_return": float(np.mean(all_returns)) if all_returns else None,
        "std_return": float(np.std(all_returns)) if all_returns else None,
        "mean_mse": float(np.mean([m.mse for m in all_metrics])) if all_metrics else None,
        "mean_mae": float(np.mean([m.mae for m in all_metrics])) if all_metrics else None,
        "mean_rmse": float(np.mean([m.rmse for m in all_metrics])) if all_metrics else None,
        "mean_psnr": float(np.mean([m.psnr for m in all_metrics])) if all_metrics else None,
        "episode_summaries": episode_summaries,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("[summary]", json.dumps(summary, indent=2))
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
