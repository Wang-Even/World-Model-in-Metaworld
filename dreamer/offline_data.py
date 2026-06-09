from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def make_offline_iterator_npz(
    path: str,
    batch_size: int,
    *,
    seed: int = 0,
    video_key: str = "videos",
    action_key: str = "actions",
    reward_key: str = "rewards",
) -> Tuple:
    """
    从本地 .npz 视频数据集中采样 batch，接口模仿 dreamer.data.make_iterator。

    期望 .npz 至少包含:
      - `video_key`: 形状 (..., T, H, W, C)，前面的维度会被合并成 batch 维 N_total
        常见情况:
          - (N, T, H, W, C)
          - (T, H, W, C)  # 单条轨迹
    可选:
      - `action_key`: 与视频同样的前导维度 (..., T, *)，缺省则填 0
      - `reward_key`: 与视频同样的前导维度 (..., T, *)，缺省则填 0
    """
    data = np.load(path, mmap_mode="r")

    if video_key not in data:
        raise KeyError(f"{video_key} not found in npz file {path}")

    vids = data[video_key]
    if vids.ndim < 4:
        raise ValueError(f"{video_key} must have at least 4 dims (..., T, H, W, C), got shape {vids.shape}")

    # 统一成 (N, T, H, W, C)
    *leading, T, H, W, C = vids.shape
    N = int(np.prod(leading)) if leading else 1
    vids = vids.reshape(N, T, H, W, C)

    # 保证 float32 且在 [0,1]（如果是 uint8）
    if vids.dtype == np.uint8:
        vids = vids.astype(np.float32) / 255.0
    else:
        vids = vids.astype(np.float32)

    def _get_or_zeros(key_name: str, dtype, fill_value=0.0):
        """
        将任意形状 (..., T, *rest) 的数组，按前导维度与视频的 leading 维对齐，
        统一 reshape 成 (N, T, *rest)。缺省则返回全 0。
        """
        if key_name in data:
            arr = data[key_name]
            # 至少要有时间维
            min_ndim = len(leading) + 1
            if arr.ndim < min_ndim:
                raise ValueError(
                    f"{key_name} must have at least {min_ndim} dims "
                    f"(..., T, ...), got {arr.shape}"
                )

            lead_a = arr.shape[: len(leading)]
            if tuple(lead_a) != tuple(leading):
                raise ValueError(
                    f"{key_name} leading dims {lead_a} != video leading dims {leading} "
                    f"in file {path}"
                )

            T_a = arr.shape[len(leading)]
            if T_a != T:
                raise ValueError(
                    f"{key_name} time dimension {T_a} != video T {T} in file {path}"
                )

            rest = arr.shape[len(leading) + 1 :]
            N_a = int(np.prod(lead_a)) if lead_a else 1
            new_shape = (N_a, T_a) + rest
            arr = arr.reshape(new_shape).astype(dtype)
        else:
            arr_shape = (N, T)
            arr = np.full(arr_shape, fill_value, dtype=dtype)
        return arr

    # NOTE: For Meta-World and other continuous-control datasets, actions are
    # real-valued. Use float32 here so that downstream models treat them as
    # continuous (e.g. ActionEncoder continuous branch) instead of discrete ids.
    actions = _get_or_zeros(action_key, np.float32, 0.0)
    rewards = _get_or_zeros(reward_key, np.float32, 0.0)

    rng_np = np.random.RandomState(seed)

    def _next_batch(key):
        # key 仅用于接口兼容，实际用 numpy 随机数
        del key
        idx = rng_np.randint(0, N, size=(batch_size,))
        v = jnp.asarray(vids[idx])      # (B, T, H, W, C)
        a = jnp.asarray(actions[idx])   # (B, T, ...)
        r = jnp.asarray(rewards[idx])   # (B, T, ...)
        new_key = jax.random.PRNGKey(rng_np.randint(0, 2**31 - 1))
        return new_key, (v, a, r)

    return _next_batch, (T, H, W, C)


def make_offline_iterator_npz_multi(
    paths: Sequence[str],
    batch_size: int,
    *,
    seed: int = 0,
    video_key: str = "videos",
    action_key: str = "actions",
    reward_key: str = "rewards",
) -> Tuple:
    """
    从多个 .npz 文件组成的数据集中采样 batch。

    每个 .npz 文件的格式要求与 make_offline_iterator_npz 相同。
    会在加载时把所有文件沿第 0 维拼接在一起：
      videos_all:  (N_total, T, H, W, C)
    """
    paths = list(paths)
    if not paths:
        raise ValueError("make_offline_iterator_npz_multi: paths 为空")

    videos_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    rewards_list: List[np.ndarray] = []

    T = H = W = C = None

    for p in paths:
        data = np.load(p, mmap_mode="r")

        if video_key not in data:
            raise KeyError(f"{video_key} not found in npz file {p}")

        vids = data[video_key]
        if vids.ndim < 4:
            raise ValueError(
                f"{video_key} in file {p} must have at least 4 dims (..., T, H, W, C), got {vids.shape}"
            )

        # 统一成 (N, T, H, W, C)
        *leading, T_p, H_p, W_p, C_p = vids.shape
        N_p = int(np.prod(leading)) if leading else 1
        vids = vids.reshape(N_p, T_p, H_p, W_p, C_p)

        if vids.dtype == np.uint8:
            vids = vids.astype(np.float32) / 255.0
        else:
            vids = vids.astype(np.float32)

        if T is None:
            # 初始化公共形状
            T, H, W, C = T_p, H_p, W_p, C_p
        else:
            # 简单检查 T,H,W,C 一致
            if (T_p, H_p, W_p, C_p) != (T, H, W, C):
                raise ValueError(
                    f"文件 {p} 的视频形状 {(T_p, H_p, W_p, C_p)} 与之前的不一致 {(T, H, W, C)}"
                )

        videos_list.append(vids)

        # Actions / rewards：形状前导维度与视频一致，缺省则补 0。
        def _get_or_zeros(key_name: str, dtype, fill_value=0.0):
            """
            与单文件版本类似：假设数组形状为 (..., T_p, *rest)，前导维与视频 leading 一致，
            统一 reshape 成 (N_p, T_p, *rest)。缺省则返回全 0。
            """
            if key_name in data:
                arr = data[key_name]
                min_ndim = len(leading) + 1
                if arr.ndim < min_ndim:
                    raise ValueError(
                        f"{key_name} in file {p} must have at least {min_ndim} dims "
                        f"(..., T, ...), got {arr.shape}"
                    )

                lead_a = arr.shape[: len(leading)]
                if tuple(lead_a) != tuple(leading):
                    raise ValueError(
                        f"{key_name} leading dims {lead_a} != video leading dims {leading} "
                        f"in file {p}"
                    )

                T_a = arr.shape[len(leading)]
                if T_a != T_p:
                    raise ValueError(
                        f"{key_name} time dimension {T_a} != video T {T_p} in file {p}"
                    )

                rest = arr.shape[len(leading) + 1 :]
                new_shape = (N_p, T_p) + rest
                arr = arr.reshape(new_shape).astype(dtype)
            else:
                arr_shape = (N_p, T_p)
                arr = np.full(arr_shape, fill_value, dtype=dtype)
            return arr

        # Same as single-file loader: treat actions as continuous float32 so
        # that the world-model sees real-valued controls rather than integer ids.
        actions_list.append(_get_or_zeros(action_key, np.float32, 0.0))
        rewards_list.append(_get_or_zeros(reward_key, np.float32, 0.0))

    videos_all = np.concatenate(videos_list, axis=0)
    actions_all = np.concatenate(actions_list, axis=0)
    rewards_all = np.concatenate(rewards_list, axis=0)

    N_total = videos_all.shape[0]
    rng_np = np.random.RandomState(seed)

    def _next_batch(key):
        del key
        idx = rng_np.randint(0, N_total, size=(batch_size,))
        v = jnp.asarray(videos_all[idx])
        a = jnp.asarray(actions_all[idx])
        r = jnp.asarray(rewards_all[idx])
        new_key = jax.random.PRNGKey(rng_np.randint(0, 2**31 - 1))
        return new_key, (v, a, r)

    return _next_batch, (T, H, W, C)