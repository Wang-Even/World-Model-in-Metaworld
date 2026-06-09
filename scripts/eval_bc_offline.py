from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from dreamer.utils import make_manager
from scripts.train_policy import RLConfig, initialize_models, encode_frames_to_spatial


def _build_cfg_from_bc_meta(
    bc_rew_ckpt: str,
    *,
    context_len: int,
) -> RLConfig:
    m_meta = make_manager(bc_rew_ckpt, item_names=("meta",))
    latest = m_meta.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No BC/rew checkpoint meta found in {bc_rew_ckpt}")
    restored = m_meta.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored.meta
    bc_cfg = meta.get("cfg") or {}
    rl_field_names = {f.name for f in fields(RLConfig)}
    cfg_kwargs = {
        name: bc_cfg[name]
        for name in rl_field_names
        if name in bc_cfg and name != "bc_rew_ckpt"
    }
    cfg_kwargs["run_name"] = bc_cfg.get("run_name") or Path(bc_rew_ckpt).resolve().parent.name
    cfg_kwargs["bc_rew_ckpt"] = bc_rew_ckpt
    cfg = RLConfig(**cfg_kwargs)
    cfg = replace(cfg, B=1, T=max(int(context_len), 2), context_length=int(context_len))
    return cfg


def _load_models(cfg: RLConfig):
    frames_init = jnp.zeros((cfg.B, cfg.T, cfg.H, cfg.W, cfg.C), dtype=jnp.float32)
    actions_init = jnp.zeros((cfg.B, cfg.T, cfg.action_dim), dtype=jnp.float32)
    return initialize_models(cfg, frames_init, actions_init)


def _episode_metrics(train_state, cfg: RLConfig, frames: np.ndarray, actions: np.ndarray):
    frames_j = jnp.asarray(frames[None, ...], dtype=jnp.float32)
    actions_j = jnp.asarray(actions[None, ...], dtype=jnp.float32)
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    z_ctx = encode_frames_to_spatial(
        frames_j,
        encoder=train_state.encoder,
        enc_vars=train_state.enc_vars,
        mae_eval_key=train_state.mae_eval_key,
        patch=cfg.patch,
        n_spatial=n_spatial,
        packing_factor=cfg.packing_factor,
    )  # (1, T, n_spatial, d_spatial)

    B, T_seq = z_ctx.shape[:2]
    task_ids = jnp.zeros((B,), dtype=jnp.int32)
    agent_tokens = train_state.task_embedder.apply(
        train_state.task_vars,
        task_ids,
        B,
        T_seq,
    )
    step_idx = jnp.full((B, T_seq), jnp.int32(np.log2(cfg.k_max)), dtype=jnp.int32)
    signal_idx = jnp.full((B, T_seq), jnp.int32(cfg.k_max - 1), dtype=jnp.int32)

    # Path 1: aligned with BC training semantics h_t -> a_{t+1}.
    _, h_train = train_state.dynamics.apply(
        train_state.dyn_vars,
        actions_j,
        step_idx,
        signal_idx,
        z_ctx,
        agent_tokens=agent_tokens,
        deterministic=True,
    )
    h_train = jnp.mean(h_train, axis=2)
    pi_train = train_state.policy_head_bc.apply(
        train_state.pi_bc_vars,
        h_train,
        deterministic=True,
    )  # (1, T, L, A)
    pred_train = np.asarray(pi_train[0, :-1, 0, :], dtype=np.float32)
    tgt_train = np.asarray(actions_j[0, 1:, :], dtype=np.float32)

    # Path 2: aligned with current inference semantics frame_t + prev action -> a_t.
    actions_shift = jnp.zeros_like(actions_j)
    actions_shift = actions_shift.at[:, 1:, :].set(actions_j[:, :-1, :])
    _, h_infer = train_state.dynamics.apply(
        train_state.dyn_vars,
        actions_shift,
        step_idx,
        signal_idx,
        z_ctx,
        agent_tokens=agent_tokens,
        deterministic=True,
    )
    h_infer = jnp.mean(h_infer, axis=2)
    pi_infer = train_state.policy_head_bc.apply(
        train_state.pi_bc_vars,
        h_infer,
        deterministic=True,
    )
    pred_infer = np.asarray(pi_infer[0, :-1, 0, :], dtype=np.float32)
    tgt_infer = np.asarray(actions_j[0, :-1, :], dtype=np.float32)

    def _metrics(pred: np.ndarray, tgt: np.ndarray):
        diff = pred - tgt
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(mse))
        return mse, mae, rmse

    train_mse, train_mae, train_rmse = _metrics(pred_train, tgt_train)
    infer_mse, infer_mae, infer_rmse = _metrics(pred_infer, tgt_infer)

    return {
        "train_path": {
            "mse": train_mse,
            "mae": train_mae,
            "rmse": train_rmse,
        },
        "infer_path": {
            "mse": infer_mse,
            "mae": infer_mae,
            "rmse": infer_rmse,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BC-head diagnostic on real dataset trajectories.")
    parser.add_argument("--bc_rew_ckpt", type=str, required=True, help="BC/reward checkpoint directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing trajectory npz files.")
    parser.add_argument("--context_len", type=int, default=8, help="Context length used for shape init.")
    parser.add_argument("--max_episodes", type=int, default=16, help="How many trajectories to evaluate.")
    parser.add_argument("--max_time_steps", type=int, default=64, help="How many leading time steps per trajectory to evaluate.")
    args = parser.parse_args()

    cfg = _build_cfg_from_bc_meta(args.bc_rew_ckpt, context_len=args.context_len)
    train_state = _load_models(cfg)

    paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No npz files found in {args.data_dir}")
    paths = paths[: args.max_episodes]

    train_mses, train_maes, train_rmses = [], [], []
    infer_mses, infer_maes, infer_rmses = [], [], []

    for idx, path in enumerate(paths):
        data = np.load(path)
        T_use = min(int(args.max_time_steps), int(data["image"].shape[0]))
        frames = data["image"][:T_use].astype(np.float32) / 255.0
        actions = data["action"][:T_use].astype(np.float32)
        metrics = _episode_metrics(train_state, cfg, frames, actions)
        tp = metrics["train_path"]
        ip = metrics["infer_path"]
        train_mses.append(tp["mse"])
        train_maes.append(tp["mae"])
        train_rmses.append(tp["rmse"])
        infer_mses.append(ip["mse"])
        infer_maes.append(ip["mae"])
        infer_rmses.append(ip["rmse"])
        print(
            f"[episode {idx:03d}] "
            f"train_path mse={tp['mse']:.6f} mae={tp['mae']:.6f} rmse={tp['rmse']:.6f} | "
            f"infer_path mse={ip['mse']:.6f} mae={ip['mae']:.6f} rmse={ip['rmse']:.6f}"
        )

    def _summary(xs):
        arr = np.asarray(xs, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

    train_mse_mean, train_mse_std = _summary(train_mses)
    train_mae_mean, train_mae_std = _summary(train_maes)
    train_rmse_mean, train_rmse_std = _summary(train_rmses)
    infer_mse_mean, infer_mse_std = _summary(infer_mses)
    infer_mae_mean, infer_mae_std = _summary(infer_maes)
    infer_rmse_mean, infer_rmse_std = _summary(infer_rmses)

    print("[summary] training-path h_t -> a_{t+1}")
    print(
        f"  mse={train_mse_mean:.6f} ± {train_mse_std:.6f}, "
        f"mae={train_mae_mean:.6f} ± {train_mae_std:.6f}, "
        f"rmse={train_rmse_mean:.6f} ± {train_rmse_std:.6f}"
    )
    print("[summary] inference-path frame_t -> a_t")
    print(
        f"  mse={infer_mse_mean:.6f} ± {infer_mse_std:.6f}, "
        f"mae={infer_mae_mean:.6f} ± {infer_mae_std:.6f}, "
        f"rmse={infer_rmse_mean:.6f} ± {infer_rmse_std:.6f}"
    )


if __name__ == "__main__":
    main()
