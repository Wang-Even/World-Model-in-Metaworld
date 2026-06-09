from functools import partial
import argparse
import os
import numpy as np

# Avoid aggressive GPU memory preallocation during module import unless user overrides it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from dreamer.models import Encoder, Decoder
from dreamer.offline_data import make_offline_iterator_npz_multi
import imageio
from pathlib import Path
from time import time
from dreamer.utils import (
    temporal_patchify,
    temporal_unpatchify,
    make_state,
    make_manager,
    try_restore,
    maybe_save,
    pack_mae_params,
    unpack_mae_params,
)


 
def init_models(rng, encoder, decoder, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)

    enc_vars = encoder.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        patch_tokens, deterministic=True
    )
    fake_z = jnp.zeros((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_z, deterministic=True
    )
    return rng, enc_vars, dec_vars

# --- forward (no jit; we jit the train_step) ---
def forward_apply(encoder, decoder, enc_vars, dec_vars, patches_btnd, *, mae_key, drop_key, train: bool):
    # Avoid TracerBool issues: pass a python bool here OR replace with lax.cond if needed.
    rngs_enc = {"mae": mae_key} if not train else {"mae": mae_key, "dropout": drop_key}
    z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd, rngs=rngs_enc, deterministic=not train)

    rngs_dec = {} if not train else {"dropout": drop_key}
    pred_btnd = decoder.apply(dec_vars, z_btLd, rngs=rngs_dec, deterministic=not train)
    return pred_btnd, mae_info  # mae_info = (mae_mask, keep_prob)

# --- loss ---
def recon_loss_from_mae(pred_btnd, patches_btnd, mae_mask):
    masked_pred   = jnp.where(mae_mask, pred_btnd, 0.0)
    masked_target = jnp.where(mae_mask, patches_btnd, 0.0)
    num = jnp.maximum(mae_mask.sum(), 1)
    return jnp.sum((masked_pred - masked_target) ** 2) / (num * pred_btnd.shape[-1])

# --- instantiate once (top-level / main) ---
lpips_loss_fn = None


def make_summary_writer(log_dir: Path):
    try:
        from tensorboardX import SummaryWriter as _SummaryWriter

        return _SummaryWriter(logdir=str(log_dir)), "tensorboardX"
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

            return _SummaryWriter(log_dir=str(log_dir)), "torch"
        except ImportError as e:
            raise ImportError(
                "No TensorBoard writer backend found. Install one of:\n"
                "  pip install tensorboardX tensorboard\n"
                "or\n"
                "  pip install torch tensorboard"
            ) from e


def maybe_print_oom_hint(exc: Exception):
    msg = str(exc)
    if "CUDA_ERROR_OUT_OF_MEMORY" in msg or "RESOURCE_EXHAUSTED" in msg:
        print("\n[OOM Hint] JAX ran out of GPU memory.")
        print("[OOM Hint] Retry with smaller batch and disable GPU preallocation:")
        print("  export XLA_PYTHON_CLIENT_PREALLOCATE=false")
        print("  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5")
        print("  python scripts/train_tokenizer.py --batch_size 8 ...")

def lpips_on_mae_recon(
    pred, target, mae_mask, *, H, W, C, patch,
    subsample_frac: float = 1.0
):
    """
    pred:    (B,T,Np,D)
    target:  (B,T,Np,D)
    mae_mask:     (B,T,Np,1)  True where patch is masked (must reconstruct)
    Returns scalar LPIPS averaged over (B,T).
    """
    # 1) Blend GT for visible patches => "recon_masked"
    recon_masked_btnd = jnp.where(mae_mask, pred, target)

    # 2) Unpatchify to (B,T,H,W,C) in [0,1]
    recon_imgs = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    target_imgs = temporal_unpatchify(target,        H, W, C, patch)

    # 3) Optional subsample frames over T to save compute
    if subsample_frac < 1.0:
        B, T = recon_imgs.shape[:2]
        step = max(1, int(1.0/subsample_frac))
        idx = jnp.arange(T)[::step]
        recon_imgs = recon_imgs[:, idx]
        target_imgs = target_imgs[:, idx]

    # 4) Rescale to [-1,1] for LPIPS
    recon_lp = jnp.clip(recon_imgs * 2.0 - 1.0, -1.0, 1.0)
    target_lp = jnp.clip(target_imgs * 2.0 - 1.0, -1.0, 1.0)

    # 5) Flatten B,T for a single LPIPS call: (B*T,H,W,C)
    BT = recon_lp.shape[0] * recon_lp.shape[1]
    H_, W_, C_ = recon_lp.shape[2], recon_lp.shape[3], recon_lp.shape[4]
    recon_lp = recon_lp.reshape((BT, H_, W_, C_))
    target_lp = target_lp.reshape((BT, H_, W_, C_))

    # 6) LPIPS returns per-example loss; average it
    lp = lpips_loss_fn(recon_lp, target_lp)  # shape (BT,)
    return jnp.mean(lp)

# --- viz step ---
@partial(jax.jit, static_argnames=("encoder","decoder","patch"))
def viz_step(encoder, decoder, enc_vars, dec_vars, batch, *, patch, mae_key, drop_key):
    # Same preprocessing as train
    patches_btnd = temporal_patchify(batch, patch)  # (B,T,Np,D)

    # Run full model (no dropout during viz)
    pred_btnd, (mae_mask_btNp1, keep_prob_bt1) = forward_apply(
        encoder, decoder, enc_vars, dec_vars, patches_btnd,
        mae_key=mae_key, drop_key=drop_key, train=False
    )

    # Compose standard MAE visualization:
    # - masked_input: what the model actually sees (masked patches)
    # - recon_masked: inpaint only masked patches (visible patches kept as GT)
    masked_input_btnd  = jnp.where(mae_mask_btNp1, 0.0, patches_btnd)
    recon_masked_btnd  = jnp.where(mae_mask_btNp1, pred_btnd, patches_btnd)
    recon_full_btnd    = pred_btnd  # decoder everywhere

    return {
        "target": patches_btnd,
        "masked_input": masked_input_btnd,
        "recon_masked": recon_masked_btnd,
        "recon_full": recon_full_btnd,
        "mae_mask": mae_mask_btNp1,
        "keep_prob": keep_prob_bt1,
    }


# --- train step ---
@partial(jax.jit, static_argnames=("encoder","decoder","tx","patch","H","W","C", "lpips_weight", "lpips_frac"))
def train_step(encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch, *,
               patch, H, W, C, master_key, step, lpips_weight=0.2, lpips_frac=1.0):
    """
    (master_key, params, opt_state, model_state, batch)
        │
        ▼
    [ compute grads ]
        │
        ▼
    Optax: (grads, opt_state, params) → (updates, new_opt_state)
    Flax:  params ⟶ apply updates → new_params
        │
        ▼
    return (new_params, new_opt_state, new_model_state, metrics)
    """
    # 1) Prepare data
    patches_btnd = temporal_patchify(batch, patch)  # (B,T,Np,Dp)

    # 2) Make per-step RNGs (fold_in ensures different masks per step even if base key repeats)
    step_key  = jax.random.fold_in(master_key, step)
    mae_key, drop_key = jax.random.split(step_key)

    # 3) Define loss fn (closes over encoder/decoder + non-param states)
    def loss_fn(packed_params):
        # Replace params in vars
        ev, dv = unpack_mae_params(packed_params, enc_vars, dec_vars)
        pred, mae_info = forward_apply(encoder, decoder, ev, dv, patches_btnd,
                                       mae_key=mae_key, drop_key=drop_key, train=True)
        mae_mask, keep_prob = mae_info
        mse = recon_loss_from_mae(pred, patches_btnd, mae_mask)

        # LPIPS on recon_masked vs target (unpatchified frames)
        if lpips_weight > 0.0:
            lpips = lpips_on_mae_recon(
                pred, patches_btnd, mae_mask,
                H=H, W=W, C=C, patch=patch, subsample_frac=lpips_frac
            )
            total = mse + lpips_weight * lpips
        else:
            lpips = 0.0
            total = mse

        aux = {
            "loss_total": total,
            "loss_mse": mse,
            "loss_lpips": lpips,
            "keep_prob": keep_prob,
        }

        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # 4) Update
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 5) Put params back into variables for next step
    new_enc_vars, new_dec_vars = unpack_mae_params(new_params, enc_vars, dec_vars)
    return new_params, opt_state, new_enc_vars, new_dec_vars, aux

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer on offline Meta-World video dataset (.npz).")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="包含多个 .npz 文件的数据集文件夹路径，每个 npz 需包含 videos 数组。",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="test",
        help="日志 / checkpoint 目录名（默认: test）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="训练 batch 大小（默认: 32）。",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=8,
        help="每个样本使用的时间长度 T_ctx（随机裁剪），默认 8 帧。",
    )
    parser.add_argument(
        "--lpips_weight",
        type=float,
        default=0.2,
        help="LPIPS 权重（默认: 0.2）；0 表示关闭 LPIPS，仅使用 MSE。",
    )
    parser.add_argument(
        "--lpips_frac",
        type=float,
        default=0.5,
        help="LPIPS 计算时采样的时间帧比例 (0,1]，用于节省算力。",
    )
    parser.add_argument(
        "--lpips_network",
        type=str,
        default="alexnet",
        choices=["alexnet", "vgg16", "squeeze"],
        help="LPIPS backbone 网络名称。",
    )
    parser.add_argument(
        "--lpips_local_dir",
        type=str,
        default="",
        help=(
            "本地 LPIPS 权重目录。若设置，将 monkeypatch jaxlpips 的 hf_hub_download，"
            "直接从该目录读取 <network>_features.safetensors 和 <network>_lpips.safetensors。"
        ),
    )
    parser.add_argument(
        "--xla_preallocate",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="设置 XLA_PYTHON_CLIENT_PREALLOCATE；auto 表示不改动当前环境变量。",
    )
    parser.add_argument(
        "--xla_mem_fraction",
        type=float,
        default=None,
        help="设置 XLA_PYTHON_CLIENT_MEM_FRACTION（例如 0.5）。",
    )
    parser.add_argument(
        "--disable_tensorboard",
        action="store_true",
        help="关闭 TensorBoard 日志写入。",
    )
    args = parser.parse_args()

    if args.xla_preallocate != "auto":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = args.xla_preallocate
    if args.xla_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
    print(
        "[jax] XLA_PYTHON_CLIENT_PREALLOCATE="
        f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '<unset>')}, "
        "XLA_PYTHON_CLIENT_MEM_FRACTION="
        f"{os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '<unset>')}"
    )

    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    if not args.disable_tensorboard:
        tb_dir = run_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer, tb_backend = make_summary_writer(tb_dir)
        print(f"[tensorboard] backend={tb_backend}, dir={tb_dir}")

    rng = jax.random.PRNGKey(0)
    B = int(args.batch_size)
    T_ctx = int(args.context_len)

    # 从指定文件夹下的多个 .npz 文件构造离线数据迭代器
    import glob

    DATA_DIR = args.data_dir
    NPZ_GLOB = os.path.join(DATA_DIR, "*.npz")
    npz_paths = sorted(glob.glob(NPZ_GLOB))
    if not npz_paths:
        raise FileNotFoundError(f"未找到匹配 {NPZ_GLOB} 的 npz 文件")

    _next_batch, (T, H, W, C) = make_offline_iterator_npz_multi(
        npz_paths,
        batch_size=B,
        seed=0,
        video_key="image",   # 你的 npz 里是 image
        action_key="action",
        reward_key="reward",
    )

    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C

    # losses and optimization
    lpips_weight = float(args.lpips_weight)
    lpips_frac = float(args.lpips_frac)
    if lpips_weight > 0.0:
        from jaxlpips import LPIPS
        import jaxlpips.utils as jaxlpips_utils

        if args.lpips_local_dir:
            local_dir = os.path.abspath(args.lpips_local_dir)
            if not os.path.isdir(local_dir):
                raise FileNotFoundError(f"LPIPS 本地目录不存在: {local_dir}")

            def _local_hf_hub_download(*args, **kwargs):
                # 兼容 hf_hub_download 的位置参数或关键字参数调用
                if "filename" in kwargs:
                    filename = kwargs["filename"]
                elif len(args) >= 2:
                    filename = args[1]
                else:
                    raise ValueError(
                        "无法从 hf_hub_download 调用中解析 filename，"
                        f"args={args}, kwargs keys={list(kwargs.keys())}"
                    )
                path = os.path.join(local_dir, filename)
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"LPIPS 本地权重缺失: {path}. "
                        f"请确认目录 {local_dir} 下包含 {filename}"
                    )
                return path

            # jaxlpips.utils.load_model 内部会调用本符号；替换后可完全离线加载。
            jaxlpips_utils.hf_hub_download = _local_hf_hub_download
            print(f"[lpips] Using local weights directory: {local_dir}")

        lpips_loss_fn = LPIPS(pretrained_network=args.lpips_network)
        print(
            f"[lpips] Enabled: weight={lpips_weight}, frac={lpips_frac}, "
            f"network={args.lpips_network}"
        )
    else:
        print("[lpips] Disabled (lpips_weight=0.0)")

    def next_batch(rng):
        """从离线数据集中取一个 batch，并在时间维上随机裁剪到长度 T_ctx。"""
        rng, (videos, actions, rewards) = _next_batch(rng)  # (B, T_full, H, W, C)
        T_full = videos.shape[1]
        # 随机裁剪一个长度为 T_ctx 的子序列（或截断到最短）
        use_T = min(T_ctx, T_full)
        if T_full > use_T:
            rng, subkey = jax.random.split(rng)
            start = int(jax.random.randint(subkey, shape=(), minval=0, maxval=T_full - use_T + 1))
            videos = videos[:, start:start + use_T]
        else:
            videos = videos[:, :use_T]
        return rng, videos

    rng, first_batch = next_batch(rng)  # warmup，小批量序列 (B, T_ctx, H, W, C)

    # models
    enc_n_latents, enc_d_bottleneck = 16, 32
    enc_kwargs = {
        "d_model": 64, "n_latents": enc_n_latents, "n_patches": num_patches, "n_heads": 4, "depth": 8, "dropout": 0.05,
        "d_bottleneck": enc_d_bottleneck, "mae_p_min": 0.0, "mae_p_max": 0.9, "time_every": 4,
    }
    dec_kwargs = {
        "d_model": 64, "n_heads": 4, "n_patches": num_patches, "n_latents": enc_n_latents, "depth": 8,
        "d_patch": D_patch, "dropout": 0.05, "time_every": 4,
    }
    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)

    # 为了避免 init 阶段的显存爆炸，只用少量样本/时间步初始化参数
    B_init = min(B, 4)
    first_batch_small = first_batch[:B_init]  # (B_init, T_ctx, H, W, C)
    first_patches = temporal_patchify(first_batch_small, patch)
    rng, enc_vars, dec_vars = init_models(
        rng,
        encoder,
        decoder,
        first_patches,
        B_init,
        first_batch_small.shape[1],
        enc_n_latents,
        enc_d_bottleneck,
    )

    # optim
    params = pack_mae_params(enc_vars, dec_vars)
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)
    max_steps = 1_000_000_000

    # ---------- ORBAX: manager + (optional) restore ----------
    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(ckpt_dir, max_to_keep=5, save_interval_steps=10_000)

    # Build example trees for safe restore (use live shapes/dtypes).
    state_example = make_state(params, opt_state, rng, step=0)
    meta_example = {"enc_kwargs": enc_kwargs, "dec_kwargs": dec_kwargs,
                    "H": H, "W": W, "C": C, "patch": patch}

    restored = try_restore(mngr, state_example, meta_example)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        # Unpack state back to your locals
        params = r.state["params"]
        opt_state = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        # Optional: you can read r.meta here if you want to sanity-check config.

        # Rebuild enc_vars/dec_vars from params so downstream apply() uses the restored params.
        enc_vars, dec_vars = unpack_mae_params(params, enc_vars, dec_vars)
        print(f"Restored checkpoint at step {latest_step} from {ckpt_dir}")

    # ---------- Train loop ----------
    last_step = start_step
    try:
        for step in range(start_step, max_steps):
            # use a fixed batch for debugging
            # _, batch = next_batch(jax.random.PRNGKey(0))
            data_start_t = time()
            rng, batch = next_batch(rng)
            data_t = time() - data_start_t
            train_start_t = time()
            rng, master_key = jax.random.split(rng)
            params, opt_state, enc_vars, dec_vars, aux = train_step(
                encoder, decoder, tx, params, opt_state, enc_vars, dec_vars, batch,
                patch=patch, H=H, W=W, C=C, master_key=master_key, step=step, lpips_weight=lpips_weight, lpips_frac=lpips_frac,
            )
            train_t = time() - train_start_t
            mse_loss = float(aux["loss_mse"])
            lpips_loss = float(aux["loss_lpips"])
            total_loss = float(aux["loss_total"])
            psnr = float(10.0 * jnp.log10(1.0 / jnp.maximum(mse_loss, 1e-10)))
            rmse = float(np.sqrt(mse_loss))
            keep_prob = float(jnp.mean(aux["keep_prob"]))
            total_t = data_t + train_t

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_total", total_loss, step)
                tb_writer.add_scalar("train/loss_mse", mse_loss, step)
                tb_writer.add_scalar("train/loss_lpips", lpips_loss, step)
                tb_writer.add_scalar("train/psnr", psnr, step)
                tb_writer.add_scalar("train/rmse", rmse, step)
                tb_writer.add_scalar("train/keep_prob", keep_prob, step)
                tb_writer.add_scalar("time/data_sec", data_t, step)
                tb_writer.add_scalar("time/train_sec", train_t, step)
                tb_writer.add_scalar("time/total_sec", total_t, step)

            # Log
            if step % 100 == 0:
                print(f"step {step:03d} |  total={total_loss:.6f} | rmse={rmse:.6f} | lpips={lpips_loss:.4f} | psnr={psnr:.4f} | time={total_t:.3f}s")

            # Save (async)
            state = make_state(params, opt_state, rng, step)
            maybe_save(mngr, step, state, meta_example)
            last_step = step

            # Viz: 画出一个样本的时间序列，对比原图和重建
            if step % 5000 == 0:
                rng, viz_key = jax.random.split(rng)
                mae_key, drop_key, vis_batch_key = jax.random.split(viz_key, 3)
                _, viz_batch = next_batch(vis_batch_key)  # (B, T, H, W, C)

                # 取 1 个样本、若干帧做可视化
                B_viz = 1
                T_viz = min(8, viz_batch.shape[1])
                viz_batch = viz_batch[:B_viz, :T_viz]

                out = viz_step(
                    encoder,
                    decoder,
                    enc_vars,
                    dec_vars,
                    viz_batch,
                    patch=patch,
                    mae_key=mae_key,
                    drop_key=drop_key,
                )

                # (B,T,H,W,C)
                target_bt = temporal_unpatchify(
                    out["target"], H, W, C, patch
                )
                recon_bt = temporal_unpatchify(
                    out["recon_full"], H, W, C, patch
                )

                target_bt = target_bt[:B_viz, :T_viz]
                recon_bt = recon_bt[:B_viz, :T_viz]

                # 拼成 2 行 (原图 / 重建) × T_viz 列的网格
                target_row = jnp.concatenate(
                    target_bt[0], axis=1
                )  # (H, T_viz*W, C)
                recon_row = jnp.concatenate(
                    recon_bt[0], axis=1
                )  # (H, T_viz*W, C)
                grid = jnp.concatenate([target_row, recon_row], axis=0)  # (2H, T_viz*W, C)
                grid = jnp.asarray(grid * 255.0, dtype=jnp.uint8)
                imageio.imwrite(run_dir / f"step_{step:06d}.png", grid)
                if tb_writer is not None:
                    tb_writer.add_image(
                        "viz/target_vs_recon",
                        np.asarray(grid),
                        step,
                        dataformats="HWC",
                    )
                    tb_writer.flush()
    except KeyboardInterrupt:
        print(f"\n[KeyboardInterrupt] Saving final tokenizer checkpoint at step {last_step} ...")
        state = make_state(params, opt_state, rng, last_step)
        save_args = ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            meta=ocp.args.JsonSave(meta_example),
        )
        mngr.save(last_step, args=save_args)
        mngr.wait_until_finished()
        print(f"[KeyboardInterrupt] Final tokenizer checkpoint saved to {ckpt_dir}")
    except Exception as e:
        print(f"\n[Exception] {e!r}")
        maybe_print_oom_hint(e)
        print(f"[Exception] Saving final tokenizer checkpoint at step {last_step} ...")
        state = make_state(params, opt_state, rng, last_step)
        save_args = ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            meta=ocp.args.JsonSave(meta_example),
        )
        mngr.save(last_step, args=save_args)
        mngr.wait_until_finished()
        print(f"[Exception] Final tokenizer checkpoint saved to {ckpt_dir}. Re-raising.")
        raise
    finally:
        # Make sure any background saves finish before exit.
        mngr.wait_until_finished()
        if tb_writer is not None:
            tb_writer.close()
