from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
from functools import partial
import json
import time
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import imageio.v2 as imageio
import orbax.checkpoint as ocp
TBWriter = None
try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    TBWriter = _TBWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter as _TBWriter  # type: ignore
        TBWriter = _TBWriter
    except Exception:
        TBWriter = None
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from dreamer.models import Encoder, Decoder, Dynamics
from dreamer.data import make_iterator
from dreamer.offline_data import make_offline_iterator_npz_multi
from dreamer.utils import (
    temporal_patchify,
    pack_bottleneck_to_spatial,
    with_params,
    make_state, make_manager, try_restore, maybe_save,
    pack_mae_params,
)

from dreamer.sampler import SamplerConfig, sample_video

# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class RealismConfig:
    # IO / ckpt
    run_name: str
    tokenizer_ckpt: str
    log_dir: str = "./logs"
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None  # if None, uses default entity
    wandb_project: str | None = None  # if None, uses run_name as project

    # data
    B: int = 64
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2
    size_min: int = 6
    size_max: int = 14
    hold_min: int = 4
    hold_max: int = 9
    diversify_data: bool = True

    # tokenizer / dynamics config
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 8
    n_heads: int = 4
    packing_factor: int = 2
    n_register: int = 4 # number of register tokens for dynamics
    n_agent: int = 1 # number of agent tokens for dynamics
    agent_space_mode: str = "wm_agent_isolated"

    # schedule
    k_max: int = 8
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap masked out
    self_fraction: float = 0.25   # used once we pass bootstrap_start

    # train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely
    write_mp4: bool = False          # if False, skip mp4 generation and export frame PNGs only
    eval_frame_stride: int = 1       # save every N-th frame from eval grids
    eval_max_frames: int = 64        # cap exported frame count per eval tag
    tb_log_video: bool = False       # if False, TensorBoard logs image sequence instead of add_video

    # offline dataset (Meta-World) config
    # 如果 data_dir 为 None，则使用合成小方块环境；否则使用指定目录下的 npz 数据集。
    data_dir: str | None = None
    video_key: str = "image"
    action_key: str = "action"
    reward_key: str = "reward"
    # 用于 dynamics 训练和 eval 的时间长度（从完整轨迹中裁剪）
    context_len: int = 8

# ---------------------------
# Small helpers
# ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    return np.concatenate(imgs_hwC, axis=1)

def _tile_videos(trip_list_hwC: list[np.ndarray], *, ncols: int = 2, pad_color: int = 0) -> np.ndarray:
    if len(trip_list_hwC) == 0:
        raise ValueError("Empty video list")
    H, W3, C = trip_list_hwC[0].shape
    B = len(trip_list_hwC)
    nrows = math.ceil(B / ncols)
    total = nrows * ncols
    if total > B:
        blank = np.full((H, W3, C), pad_color, dtype=trip_list_hwC[0].dtype)
        trip_list_hwC = trip_list_hwC + [blank] * (total - B)
    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = trip_list_hwC[idx:idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid

# ---------------------------
# Tokenizer restore (uses your Orbax layout & utils)
# ---------------------------

def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta

# ---------------------------
# Single efficient training step (always used)
# ---------------------------

@partial(
    jax.jit,
    static_argnames=("encoder","dynamics","tx","patch","n_spatial","k_max","packing_factor","B","T","B_self"),
)
def train_step_efficient(
    encoder, dynamics, tx,
    params, opt_state,
    enc_vars, dyn_vars,
    frames, actions,
    *,
    patch: int,
    B: int, T: int, B_self: int,            # assume 0 <= B_self < B
    n_spatial: int, k_max: int, packing_factor: int,
    master_key: jnp.ndarray, step: int, bootstrap_start: int,
):
    """
    Deterministic two-branch training (one fused main forward):
      - first B_emp rows: empirical flow at d_min = 1/k_max
      - last  B_self rows: bootstrap self-consistency with d > d_min
    If step < bootstrap_start, the bootstrap contribution is masked to 0 (but we still
    execute one fused path to keep a single jit and stable shapes).
    """
    @partial(jax.jit, static_argnames=("shape_bt","k_max",))
    def _sample_tau_for_step(rng, shape_bt, k_max:int, step_idx:jnp.ndarray, *, dtype=jnp.float32):
        B_, T_ = shape_bt
        K = (1 << step_idx)
        u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
        j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
        tau = j_idx.astype(dtype) / K.astype(dtype)
        tau_idx = j_idx * (k_max // K)
        return tau, tau_idx

    @partial(jax.jit, static_argnames=("shape_bt","k_max",))
    def _sample_step_excluding_dmin(rng, shape_bt, k_max:int):
        B_, T_ = shape_bt
        emax = jnp.log2(k_max).astype(jnp.int32)
        step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)  # exclude emax
        d = 1.0 / (1 << step_idx).astype(jnp.float32)
        return d, step_idx

    # ---------- Param-free precompute ----------
    patches_btnd = temporal_patchify(frames, patch)

    # RNGs
    step_key = jax.random.fold_in(master_key, step)
    enc_key, key_sigma_full, key_step_self, key_noise_full, drop_key = jax.random.split(step_key, 5)

    # Frozen encoder → spatial tokens (clean target z1)
    z_bottleneck, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": enc_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_bottleneck, n_spatial=n_spatial, k=packing_factor)  # (B,T,Sz,Dz)

    # Deterministic batch split
    B_emp  = B - B_self
    actions_full = actions
    emax = jnp.log2(k_max).astype(jnp.int32)

    # --- Step indices (encode d) ---
    step_idx_emp  = jnp.full((B_emp,  T), emax, dtype=jnp.int32)             # d = d_min
    # If B_self == 0, create a dummy 0xT array – slicing below handles it.
    d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), k_max)
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)   # (B,T)

    # --- Signal levels on each row's grid (one call for whole batch) ---
    sigma_full, sigma_idx_full = _sample_tau_for_step(key_sigma_full, (B, T), k_max, step_idx_full)
    sigma_emp   = sigma_full[:B_emp]
    sigma_self  = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 ---
    z0_full      = jax.random.normal(key_noise_full, z1.shape, dtype=z1.dtype)
    z_tilde_full = (1.0 - sigma_full)[...,None,None] * z0_full + sigma_full[...,None,None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    # --- Ramp weights ---
    w_emp  = 0.9 * sigma_emp  + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # --- Half-step metadata for self rows ---
    d_half            = d_self / 2.0
    step_idx_half     = step_idx_self + 1
    sigma_plus        = sigma_self + d_half
    sigma_idx_plus    = sigma_idx_self + (k_max * d_half).astype(jnp.int32)

    def loss_and_aux(p):
        local_dyn = with_params(dyn_vars, p)
        drop_main, drop_h1, drop_h2 = jax.random.split(drop_key, 3)

        # Main forward (emp + self)
        z1_hat_full, _ = dynamics.apply(
            local_dyn, actions_full, step_idx_full, sigma_idx_full, z_tilde_full,
            rngs={"dropout": drop_main}, deterministic=False,
        )  # (B,T,Sz,Dz)

        z1_hat_emp  = z1_hat_full[:B_emp]
        z1_hat_self = z1_hat_full[B_emp:]

        # Flow loss on empirical rows (to z1)
        flow_per = jnp.mean((z1_hat_emp - z1[:B_emp])**2, axis=(2,3))        # (B_emp,T)
        loss_emp = jnp.mean(flow_per * w_emp)

        # Self-consistency (bootstrap) on self rows
        # If B_self == 0, shapes are 0-sized and reductions become NaN; guard with mask.
        do_boot = (B_self > 0) & (step >= bootstrap_start)

        def _boot_loss():
            z1_hat_half1, _ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_self, z_tilde_self,
                rngs={"dropout": drop_h1}, deterministic=False,
            )
            b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            z_prime = z_tilde_self + b_prime * d_half[...,None,None]
            z1_hat_half2, _ = dynamics.apply(
                local_dyn, actions_full[B_emp:], step_idx_half, sigma_idx_plus, z_prime,
                rngs={"dropout": drop_h2}, deterministic=False,
            )
            b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - sigma_plus)[...,None,None]
            vhat_sigma = (z1_hat_self - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
            boot_per = (1.0 - sigma_self)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))  # (B_self,T)
            loss_self = jnp.mean(boot_per * w_self)
            return loss_self, jnp.mean(boot_per)

        loss_self, boot_mse = jax.lax.cond(
            do_boot,
            _boot_loss,
            lambda: (jnp.array(0.0, dtype=z1.dtype), jnp.array(0.0, dtype=z1.dtype)),
        )

        # Combine (row-weighted by nominal B parts; denominator B keeps scale constant)
        loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

        aux = {
            "loss_total": loss,
            "flow_mse": jnp.mean(flow_per),
            "bootstrap_mse": boot_mse,
        }
        return loss, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, aux

# ---------------------------
# Eval regimes & plan JSON (unchanged core logic)
# ---------------------------

def _eval_regimes_for_realism(cfg, *, ctx_length: int, H: int, W: int, C: int, patch: int, T_total: int):
    common = dict(
        k_max=cfg.k_max,
        horizon=min(32, T_total - ctx_length),
        ctx_length=ctx_length,
        ctx_signal_tau=1.0,   # was 0.99 — make context clean for fair PSNR
        H=H, W=W, C=C, patch=patch,
        n_spatial=cfg.enc_n_latents // cfg.packing_factor,
        packing_factor=cfg.packing_factor,
        start_mode="pure",
        rollout="autoregressive",
        # optional: see item 3 below
        # match_ctx_tau=False,
    )
    regs = []
    regs.append(("finest_pure_AR", SamplerConfig(schedule="finest", **common)))
    regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
    return regs


def _plan_from_sampler_conf(s: SamplerConfig) -> Dict[str, Any]:
    def _is_pow2_frac(x: float) -> bool:
        if x <= 0 or x > 1: return False
        inv = round(1.0 / x)
        return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

    if s.schedule == "finest":
        d = 1.0 / float(s.k_max)
    else:
        if s.d is None or not _is_pow2_frac(s.d):
            raise ValueError("shortcut schedule requires d = 1/(power of two)")
        if s.d < 1.0 / float(s.k_max):
            raise ValueError("d finer than finest")
        d = float(s.d)

    tau0 = 0.0
    S = int(round((1.0 - tau0) / d))
    e = int(round(np.log2(round(1.0 / d))))
    tau_seq = [round(tau0 + i*d, 6) for i in range(S + 1)]
    tau_seq[-1] = 1.0
    return dict(
        rollout=s.rollout,
        start_mode=s.start_mode,
        ctx_length=s.ctx_length,
        horizon=s.horizon,
        schedule=s.schedule,
        d=d,
        e=e,
        S=S,
        tau_seq=tau_seq,
        k_max=s.k_max,
        add_ctx_noise_std=getattr(s, "add_ctx_noise_std", 0.0),
    )

# ---------------------------
# Video building and saving utilities
# ---------------------------

def build_tiled_video_frames(
    gt_frames: jnp.ndarray,
    floor_frames: jnp.ndarray,
    pred_frames: jnp.ndarray,
    batch_size: int,
) -> list[np.ndarray]:
    """
    Build tiled video frames from ground truth, floor, and prediction frames.

    Each frame in the output contains a grid of triplets (GT | Floor | Pred) stacked horizontally,
    with multiple batch items tiled vertically/horizontally.

    Args:
        gt_frames: Ground truth frames (B, T, H, W, C)
        floor_frames: Floor/reference frames (B, T, H, W, C)
        pred_frames: Predicted frames (B, T, H, W, C)
        batch_size: Batch size B

    Returns:
        List of grid frames, one per time step
    """
    gt_np_all = _to_uint8(gt_frames)
    floor_np_all = _to_uint8(floor_frames)
    pred_np_all = _to_uint8(pred_frames)

    T_total = gt_np_all.shape[1]
    ncols = 1 if batch_size <= 2 else min(8, batch_size)
    grid_frames = []

    for t_idx in range(T_total):
        trip_list = [
            _stack_wide(gt_np_all[b, t_idx], floor_np_all[b, t_idx], pred_np_all[b, t_idx])
            for b in range(batch_size)
        ]
        grid_img = _tile_videos(trip_list, ncols=ncols, pad_color=0)
        grid_frames.append(grid_img)

    return grid_frames

def save_evaluation_video(
    grid_frames: list[np.ndarray],
    output_path: Path,
    tag: str,
) -> bool:
    """
    Save grid frames as an MP4 video file.

    Args:
        grid_frames: List of grid frames to write
        output_path: Path where MP4 should be saved
        tag: Tag for error messages

    Returns:
        True if successful, False otherwise
    """
    try:
        with imageio.get_writer(output_path, fps=25, codec="libx264", quality=8) as w:
            for fr in grid_frames:
                w.append_data(fr)
        return True
    except Exception as e:
        print(f"[eval:{tag}] MP4 write skipped ({e})")
        return False

def save_evaluation_frames(
    grid_frames: list[np.ndarray],
    output_dir: Path,
    *,
    tag: str,
    frame_stride: int = 1,
    max_frames: int = 64,
) -> int:
    """
    Save sampled grid frames as PNG files for visual comparison.

    Returns:
        Number of frames successfully written.
    """
    if not grid_frames:
        return 0

    stride = max(1, int(frame_stride))
    idx = list(range(0, len(grid_frames), stride))
    if max_frames > 0 and len(idx) > max_frames:
        pick = np.linspace(0, len(idx) - 1, max_frames, dtype=np.int32)
        idx = [idx[i] for i in pick]

    written = 0
    for t_idx in idx:
        out_path = output_dir / f"{tag}_t{t_idx:03d}.png"
        try:
            imageio.imwrite(out_path, grid_frames[t_idx])
            written += 1
        except Exception as e:
            print(f"[eval:{tag}] PNG frame write skipped at t={t_idx} ({e})")
    return written

def log_grid_frames_to_tensorboard(
    tb_writer,
    *,
    tag: str,
    step: int,
    grid_frames: list[np.ndarray],
    fps: int = 10,
    prefer_video: bool = False,
):
    """
    Log tiled evaluation frames to TensorBoard.
    Tries add_video first; if backend is unavailable, falls back to add_images.
    """
    if tb_writer is None or not grid_frames:
        return

    frames_np = np.asarray(grid_frames)  # (T, H, W, C)
    if frames_np.ndim != 4 or frames_np.shape[-1] not in (1, 3, 4):
        return

    sampled_n = min(64, frames_np.shape[0])
    idx = np.linspace(0, frames_np.shape[0] - 1, sampled_n, dtype=np.int32)
    frames_np = frames_np[idx]

    video = frames_np.astype(np.float32)
    if video.max() > 1.0:
        video = video / 255.0

    # (T,H,W,C) -> (1,T,C,H,W) for TensorBoard video
    video_n = np.transpose(video, (0, 3, 1, 2))[None, ...]
    if prefer_video:
        try:
            tb_writer.add_video(tag, video_n, global_step=step, fps=fps)
            return
        except Exception:
            pass
    # Default/fallback: show as image sequence (T,C,H,W)
    tb_writer.add_images(f"{tag}_frames", video_n[0], global_step=step)

def save_evaluation_plan(
    sampler_conf: SamplerConfig,
    step: int,
    mse: float,
    psnr: float,
    output_path: Path,
):
    """
    Save evaluation plan/metadata as JSON.

    Args:
        sampler_conf: Sampler configuration
        step: Training step number
        mse: Mean squared error
        psnr: Peak signal-to-noise ratio in dB
        output_path: Path where JSON should be saved
    """
    plan = _plan_from_sampler_conf(sampler_conf)
    plan["step"] = int(step)
    plan["mse"] = float(mse)
    plan["psnr_db"] = float(psnr)

    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)

# ---------------------------
# Meta for dynamics checkpoints
# ---------------------------

def make_dynamics_meta(
    *,
    enc_kwargs: dict,
    dec_kwargs: dict,
    dynamics_kwargs: dict,
    H: int, W: int, C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_spatial: int,
    tokenizer_ckpt_dir: str | None = None,
    cfg: Dict[str, Any] | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dec_kwargs": dec_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H, "W": W, "C": C, "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_spatial": n_spatial,
        "tokenizer_ckpt_dir": tokenizer_ckpt_dir,
        "cfg": cfg or {},
    }

# ---------------------------
# Training state dataclass
# ---------------------------

@dataclass
class TrainState:
    """Container for all training-related state (models, variables, optimizer, etc.)."""
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    enc_vars: dict
    dec_vars: dict
    dyn_vars: dict
    params: dict
    enc_kwargs: dict
    dec_kwargs: dict
    dyn_kwargs: dict
    tx: optax.Transform
    opt_state: optax.OptState
    mae_eval_key: jnp.ndarray

# ---------------------------
# Model initialization
# ---------------------------

def initialize_models_and_tokenizer(
    cfg: RealismConfig,
    frames_init: jnp.ndarray,
    actions_init: jnp.ndarray,
) -> TrainState:
    """
    Initialize encoder, decoder, dynamics models and restore tokenizer.

    Returns:
        TrainState containing all initialized models, variables, and optimizer state.
    """
    # 推断实际 H, W, C（来自数据集），不再依赖 cfg.H/W/C
    B_init, T_init, H, W, C = frames_init.shape
    patch = cfg.patch
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = cfg.k_max

    enc_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        depth=cfg.enc_depth,
        dropout=0.0,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_heads=cfg.n_heads,
        depth=cfg.dec_depth,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    n_spatial = cfg.enc_n_latents // cfg.packing_factor # number of spatial tokens for dynamics
    dyn_kwargs = dict(
        d_model=cfg.d_model_dyn,
        d_bottleneck=cfg.enc_d_bottleneck,
        d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
        n_spatial=n_spatial, n_register=cfg.n_register,
        n_heads=cfg.n_heads, depth=cfg.dyn_depth,
        space_mode=cfg.agent_space_mode, n_agent=cfg.n_agent,
        dropout=0.0, k_max=k_max, 
        time_every=4,
    )

    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)

    patches_btnd = temporal_patchify(frames_init, patch)
    rng = jax.random.PRNGKey(0)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    # decoder 初始化时 B/T 只影响占位形状，直接用当前 batch 的尺寸即可
    fake_z = jnp.zeros((B_init, T_init, cfg.enc_n_latents, cfg.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # Restore tokenizer
    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        cfg.tokenizer_ckpt, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    # Build initial z1 to shape dynamics init
    mae_eval_key = jax.random.PRNGKey(777)
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=cfg.packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    # 对 dynamics 来说，只需要 batch/time 维度与 actions_init / z1 对齐
    B_dyn, T_dyn = actions_init.shape[:2]
    step_idx = jnp.full((B_dyn, T_dyn), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((B_dyn, T_dyn), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)
    params = dyn_vars["params"]

    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    return TrainState(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        enc_vars=enc_vars,
        dec_vars=dec_vars,
        dyn_vars=dyn_vars,
        params=params,
        enc_kwargs=enc_kwargs,
        dec_kwargs=dec_kwargs,
        dyn_kwargs=dyn_kwargs,
        tx=tx,
        opt_state=opt_state,
        mae_eval_key=mae_eval_key,
    )

# ---------------------------
# Evaluation logic
# ---------------------------

def run_evaluation(
    cfg: RealismConfig,
    step: int,
    train_state: TrainState,
    next_batch,
    vis_dir: Path,
    tb_writer=None,
):
    """
    Run periodic evaluation: sample videos, compute metrics, and save visualization.

    Args:
        cfg: Configuration object
        step: Current training step
        train_state: TrainState containing all models, variables, and optimizer state
        next_batch: Data iterator function
        vis_dir: Directory for visualization outputs
    """
    val_rng = jax.random.PRNGKey(9999)
    _, (val_frames, val_actions, _) = next_batch(val_rng)
    dyn_vars_eval = with_params(train_state.dyn_vars, train_state.params)
    B_eval, T_total, H, W, C = val_frames.shape
    ctx_length = min(32, T_total - 1)
    regimes = _eval_regimes_for_realism(
        cfg,
        ctx_length=ctx_length,
        H=H,
        W=W,
        C=C,
        patch=cfg.patch,
        T_total=T_total,
    )

    for tag, sampler_conf in regimes:
        sampler_conf.mae_eval_key = train_state.mae_eval_key
        sampler_conf.rng_key = jax.random.PRNGKey(4242)
        t0 = time.time()

        pred_frames, floor_frames, gt_frames = sample_video(
            encoder=train_state.encoder,
            decoder=train_state.decoder,
            dynamics=train_state.dynamics,
            enc_vars=train_state.enc_vars,
            dec_vars=train_state.dec_vars,
            dyn_vars=dyn_vars_eval,
            frames=val_frames, actions=val_actions, config=sampler_conf,
        )

        dt = time.time() - t0
        HZ = sampler_conf.horizon
        mse = float(jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2))
        psnr = float(10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12)))
        print(f"[eval:{tag}] step={step:06d} | AR_hz={HZ} | MSE={mse:.6g} | PSNR={psnr:.2f} dB | {dt:.2f}s")

        # Build tiled video frames
        grid_frames = build_tiled_video_frames(
            gt_frames=gt_frames,
            floor_frames=floor_frames,
            pred_frames=pred_frames,
            batch_size=B_eval,
        )

        # Save video and plan
        tag_dir = _ensure_dir(vis_dir / f"step_{step:06d}")
        plan_path = tag_dir / f"{tag}_plan.json"
        frame_dir = _ensure_dir(tag_dir / "frames")
        mp4_path = tag_dir / f"{tag}_grid.mp4"

        saved_frames = save_evaluation_frames(
            grid_frames,
            frame_dir,
            tag=tag,
            frame_stride=cfg.eval_frame_stride,
            max_frames=cfg.eval_max_frames,
        )
        mp4_written = False
        if cfg.write_mp4:
            mp4_written = save_evaluation_video(grid_frames, mp4_path, tag)
        save_evaluation_plan(sampler_conf, step, mse, psnr, plan_path)
        log_grid_frames_to_tensorboard(
            tb_writer,
            tag=f"eval/{tag}/grid_video",
            step=step,
            grid_frames=grid_frames,
            fps=10,
            prefer_video=cfg.tb_log_video,
        )
        if tb_writer is not None:
            tb_writer.add_scalar(f"eval/{tag}/mse", mse, step)
            tb_writer.add_scalar(f"eval/{tag}/psnr", psnr, step)
            tb_writer.add_scalar(f"eval/{tag}/horizon", HZ, step)
            tb_writer.add_scalar(f"eval/{tag}/eval_time", dt, step)

        if cfg.write_mp4 and mp4_written:
            print(
                f"[eval:{tag}] wrote {saved_frames} frame PNGs, {mp4_path.name}, and {plan_path.name} in {tag_dir}"
            )
        else:
            print(
                f"[eval:{tag}] wrote {saved_frames} frame PNGs and {plan_path.name} in {tag_dir}"
            )

        # Log to wandb
        if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            # Log metrics
            wandb.log({
                f"eval/{tag}/mse": mse,
                f"eval/{tag}/psnr": psnr,
                f"eval/{tag}/horizon": HZ,
                f"eval/{tag}/eval_time": dt,
            }, step=step)
            if cfg.write_mp4 and mp4_written and grid_frames:
                wandb.log({
                    f"eval/{tag}/video": wandb.Video(mp4_path, format="mp4"),
                }, step=step)

# ---------------------------
# Main
# ---------------------------

def run(cfg: RealismConfig):
    # Initialize wandb if enabled
    if cfg.use_wandb:
        if not WANDB_AVAILABLE:
            print("[warning] wandb requested but not installed. Install with: pip install wandb")
            print("[warning] Continuing without wandb logging.")
        else:
            wandb_project = cfg.wandb_project or cfg.run_name
            wandb.init(
                entity=cfg.wandb_entity,
                project=wandb_project,
                name=cfg.run_name,
                config=asdict(cfg),
                dir=str(Path(cfg.log_dir).resolve()),
            )
            print(f"[wandb] Initialized run: {wandb.run.name if wandb.run else 'N/A'}")

    # Output dirs
    root = _ensure_dir(Path(cfg.log_dir))
    run_dir = _ensure_dir(root / cfg.run_name)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    tb_dir = _ensure_dir(run_dir / "tb")
    print(f"[setup] writing artifacts to: {run_dir.resolve()}")
    if TBWriter is not None:
        tb_writer = TBWriter(log_dir=str(tb_dir))
        print(f"[tensorboard] Logging to: {tb_dir}")
    else:
        tb_writer = None
        print("[tensorboard] SummaryWriter not available (install torch or tensorboardX).")

    # Data iterator
    # 如果 cfg.data_dir 为空，则使用原来的合成小方块环境；
    # 否则从 Meta-World 离线 npz 数据集采样 (image, action, reward)。
    if cfg.data_dir is None:
        next_batch = make_iterator(
            cfg.B, cfg.T, cfg.H, cfg.W, cfg.C,
            pixels_per_step=cfg.pixels_per_step,
            size_min=cfg.size_min, size_max=cfg.size_max,
            hold_min=cfg.hold_min, hold_max=cfg.hold_max,
            fg_min_color=0 if cfg.diversify_data else 128,
            fg_max_color=255 if cfg.diversify_data else 128,
            bg_min_color=0 if cfg.diversify_data else 255,
            bg_max_color=255 if cfg.diversify_data else 255,
        )
        # 对于合成数据，T_train 直接等于 cfg.T，H/W/C 也用配置值
        T_train = cfg.T
        H_data, W_data, C_data = cfg.H, cfg.W, cfg.C
    else:
        import os, glob
        NPZ_GLOB = os.path.join(cfg.data_dir, "*.npz")
        npz_paths = sorted(glob.glob(NPZ_GLOB))
        if not npz_paths:
            raise FileNotFoundError(f"[dynamics] 未在 {cfg.data_dir} 下找到匹配 {NPZ_GLOB} 的 npz 文件")

        _next_batch, (T_raw, H_data, W_data, C_data) = make_offline_iterator_npz_multi(
            npz_paths,
            batch_size=cfg.B,
            seed=0,
            video_key=cfg.video_key,
            action_key=cfg.action_key,
            reward_key=cfg.reward_key,
        )

        if cfg.context_len <= 0 or cfg.context_len > T_raw:
            raise ValueError(
                f"[dynamics] context_len={cfg.context_len} 非法，应在 1..{T_raw} 之间 "
                f"(由离线数据集时间长度 T_raw 推断)"
            )
        T_train = cfg.context_len

        def next_batch(rng):
            """从离线 Meta-World 数据集中采样 batch，并在时间维上随机裁剪到长度 T_train。

            注意：此处保留连续动作向量 (B,T,A_dim)，让 Dynamics 的 ActionEncoder
            使用连续分支（MLP）编码 action token，而不再手工离散化。
            """
            rng, (frames, actions, rewards) = _next_batch(rng)  # frames: (B, T_raw, H, W, C)
            T_full = frames.shape[1]
            if T_full > T_train:
                rng, subkey = jax.random.split(rng)
                start = int(
                    jax.random.randint(
                        subkey, shape=(), minval=0, maxval=T_full - T_train + 1
                    )
                )
                frames = frames[:, start:start + T_train]
                actions = actions[:, start:start + T_train]
                rewards = rewards[:, start:start + T_train]
            else:
                frames = frames[:, :T_train]
                actions = actions[:, :T_train]
                rewards = rewards[:, :T_train]

            return rng, (frames, actions, rewards)

    # Initialize models and restore tokenizer
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init, _) = next_batch(init_rng)

    train_state = initialize_models_and_tokenizer(cfg, frames_init, actions_init)

    # Extract some values for checkpointing
    patch = cfg.patch
    k_max = cfg.k_max
    n_spatial = cfg.enc_n_latents // cfg.packing_factor

    # -------- Orbax manager & (optional) restore --------
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    meta = make_dynamics_meta(
        enc_kwargs=train_state.enc_kwargs,
        dec_kwargs=train_state.dec_kwargs,
        dynamics_kwargs=train_state.dyn_kwargs,
        H=H_data, W=W_data, C=C_data, patch=patch,
        k_max=k_max, packing_factor=cfg.packing_factor, n_spatial=n_spatial,
        tokenizer_ckpt_dir=cfg.tokenizer_ckpt,
        cfg=asdict(cfg),
    )

    rng = jax.random.PRNGKey(0)
    state_example = make_state(train_state.params, train_state.opt_state, rng, step=0)
    restored = try_restore(mngr, state_example, meta)

    start_step = 0
    if restored is not None:
        latest_step, r = restored
        train_state.params = r.state["params"]
        train_state.opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"]) + 1
        train_state.dyn_vars = with_params(train_state.dyn_vars, train_state.params)
        print(f"[restore] Resumed from {ckpt_dir} at step={latest_step}")

    # -------- Training loop --------
    train_rng = jax.random.PRNGKey(2025)
    data_rng = jax.random.PRNGKey(12345)

    start_wall = time.time()
    last_step = start_step
    try:
        for step in range(start_step, cfg.max_steps + 1):
            # Data
            data_rng, batch_key = jax.random.split(data_rng)
            _, (frames, actions, _) = next_batch(batch_key)

            # RNG for this step
            train_rng, master_key = jax.random.split(train_rng)

            # Decide current B_self based on warm-up (static arg requires a single value; we keep B_self fixed
            # and gate its contribution inside the jit via bootstrap_start masking).
            B_self = max(0, int(round(cfg.self_fraction * cfg.B)))

            train_step_start_time = time.time()
            train_state.params, train_state.opt_state, aux = train_step_efficient(
                train_state.encoder, train_state.dynamics, train_state.tx,
                train_state.params, train_state.opt_state,
                train_state.enc_vars, train_state.dyn_vars,
                frames, actions,
                patch=cfg.patch, B=cfg.B, T=T_train, B_self=B_self,
                n_spatial=n_spatial, k_max=k_max, packing_factor=cfg.packing_factor,
                master_key=master_key, step=step, bootstrap_start=cfg.bootstrap_start,
            )

            # Logging
            if (step % cfg.log_every == 0) or (step == cfg.max_steps):
                total_loss = float(aux["loss_total"])
                flow_mse = float(aux['flow_mse'])
                boot_mse = float(aux['bootstrap_mse'])
                step_time = time.time() - train_step_start_time
                total_time = time.time() - start_wall
                steps_per_sec = 1.0 / max(step_time, 1e-9)
                self_fraction_effective = float(B_self / max(cfg.B, 1))

                pieces = [
                    f"[train] step={step:06d}",
                    f"loss={total_loss:.6g}",
                    f"flow_mse={flow_mse:.6g}",
                    f"boot_mse={boot_mse:.6g}",
                    f"t={step_time:.4f}s",
                    f"total_t={total_time:.1f}s",
                ]
                print(" | ".join(pieces))

                # Log to wandb
                if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/loss_total": total_loss,
                        "train/flow_mse": flow_mse,
                        "train/bootstrap_mse": boot_mse,
                        "train/lr": cfg.lr,
                        "train/self_fraction_effective": self_fraction_effective,
                        "train/steps_per_sec": steps_per_sec,
                        "train/step_time": step_time,
                        "train/total_time": total_time,
                        "step": step,
                    }, step=step)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss_total", total_loss, step)
                    tb_writer.add_scalar("train/flow_mse", flow_mse, step)
                    tb_writer.add_scalar("train/bootstrap_mse", boot_mse, step)
                    tb_writer.add_scalar("train/lr", cfg.lr, step)
                    tb_writer.add_scalar("train/self_fraction_effective", self_fraction_effective, step)
                    tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, step)
                    tb_writer.add_scalar("train/step_time", step_time, step)
                    tb_writer.add_scalar("train/total_time", total_time, step)

            # Save (async) when policy says we should
            state = make_state(train_state.params, train_state.opt_state, train_rng, step)
            maybe_save(mngr, step, state, meta)
            last_step = step

            # Periodic lightweight AR eval
            if cfg.write_video_every and (step % cfg.write_video_every == 0):
                run_evaluation(
                    cfg=cfg,
                    step=step,
                    train_state=train_state,
                    next_batch=next_batch,
                    vis_dir=vis_dir,
                    tb_writer=tb_writer,
                )
    except KeyboardInterrupt:
        print(f"\n[KeyboardInterrupt] Saving final dynamics checkpoint at step {last_step} ...")
        state = make_state(train_state.params, train_state.opt_state, train_rng, last_step)
        save_args = ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            meta=ocp.args.JsonSave(meta),
        )
        mngr.save(last_step, args=save_args)
        mngr.wait_until_finished()
        print(f"[KeyboardInterrupt] Final dynamics checkpoint saved to {ckpt_dir}")
    except Exception as e:
        print(f"\n[Exception] {e!r}")
        print(f"[Exception] Saving final dynamics checkpoint at step {last_step} ...")
        state = make_state(train_state.params, train_state.opt_state, train_rng, last_step)
        save_args = ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            meta=ocp.args.JsonSave(meta),
        )
        mngr.save(last_step, args=save_args)
        mngr.wait_until_finished()
        print(f"[Exception] Final dynamics checkpoint saved to {ckpt_dir}. Re-raising.")
        raise
    finally:
        # Ensure all writes finished
        mngr.wait_until_finished()
        if tb_writer is not None:
            tb_writer.close()

    # Save final config
    (run_dir / "config.txt").write_text("\n".join([f"{k}={v}" for k, v in asdict(cfg).items()]))

    # Finish wandb run
    if cfg.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("[wandb] Finished logging.")


if __name__ == "__main__":
    cfg = RealismConfig(
    run_name="dynamics_button_press",
    tokenizer_ckpt="/home/ywwang/dreamer4-jax/logs/tokenizer/checkpoints",
    log_dir="/home/ywwang/dreamer4-jax/logs",
    use_wandb=False,
    max_steps=1000000,
    log_every=5000,
    lr=3e-4,
    write_video_every=50000,
    ckpt_save_every=50000,
    ckpt_max_to_keep=2,
    B=32,
    context_len=8,
    data_dir="/home/ywwang/dreamer4-jax/metaworld_10tasks_200eps/button_press/eval_eps",
)
    print("Running realism config:\n  " + "\n  ".join([f"{k}={v}" for k,v in asdict(cfg).items()]))
    run(cfg)
