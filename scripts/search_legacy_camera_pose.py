from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from dreamer.veorl_bridge import VeoRLBridgeEnv


def _metrics(img: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    diff = img.astype(np.float32) - target.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


def _render_candidate(
    *,
    env_name: str,
    bridge_python: str,
    camera_name: str,
    camera_pos: tuple[float, float, float],
    camera_fovy: float,
    render_width: int,
    render_height: int,
    rotate_180: bool,
) -> tuple[np.ndarray, np.ndarray]:
    env = VeoRLBridgeEnv(
        env_name,
        camera_name=camera_name,
        camera_pos=camera_pos,
        camera_fovy=camera_fovy,
        render_width=render_width,
        render_height=render_height,
        python_bin=bridge_python,
    )
    obs = env.reset()
    env.close()
    raw = np.asarray(obs["image_ori"])
    img = raw
    if rotate_180:
        img = np.rot90(img, 2)
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return raw, small


def main() -> None:
    parser = argparse.ArgumentParser(description="Search legacy corner2 camera pose/fovy against dataset frames.")
    parser.add_argument("--env", type=str, default="button-press-v3")
    parser.add_argument("--bridge_python", type=str, default=".conda-veorl/bin/python")
    parser.add_argument("--camera_name", type=str, default="corner2")
    parser.add_argument("--render_width", type=int, default=128)
    parser.add_argument("--render_height", type=int, default=128)
    parser.add_argument("--rotate_180", action="store_true")
    parser.add_argument("--dataset_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="logs/legacy_camera_pose_search")
    args = parser.parse_args()

    target = np.load(args.dataset_npz)["image"][0]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    coarse_center = np.array([0.75, 0.075, 0.7], dtype=np.float32)
    coarse_fovys = [35.0, 40.0, 45.0, 50.0, 55.0]
    coarse_x = np.arange(0.60, 0.91, 0.05)
    coarse_y = np.arange(-0.05, 0.201, 0.05)
    coarse_z = np.arange(0.55, 0.91, 0.05)

    coarse_results = []
    for x in coarse_x:
        for y in coarse_y:
            for z in coarse_z:
                for fovy in coarse_fovys:
                    pos = (float(x), float(y), float(z))
                    raw, small = _render_candidate(
                        env_name=args.env,
                        bridge_python=args.bridge_python,
                        camera_name=args.camera_name,
                        camera_pos=pos,
                        camera_fovy=float(fovy),
                        render_width=args.render_width,
                        render_height=args.render_height,
                        rotate_180=args.rotate_180,
                    )
                    mae, mse, rmse = _metrics(small, target)
                    coarse_results.append((mae, mse, rmse, pos, float(fovy), raw, small))
                    print("coarse", pos, fovy, mae, rmse, flush=True)

    coarse_results.sort(key=lambda x: x[0])
    best_coarse = coarse_results[0]

    cx, cy, cz = best_coarse[3]
    cf = best_coarse[4]
    fine_results = []
    fine_x = np.arange(cx - 0.04, cx + 0.041, 0.01)
    fine_y = np.arange(cy - 0.04, cy + 0.041, 0.01)
    fine_z = np.arange(cz - 0.04, cz + 0.041, 0.01)
    fine_fovys = np.arange(cf - 6.0, cf + 6.1, 1.0)

    for x in fine_x:
        for y in fine_y:
            for z in fine_z:
                for fovy in fine_fovys:
                    pos = (float(x), float(y), float(z))
                    raw, small = _render_candidate(
                        env_name=args.env,
                        bridge_python=args.bridge_python,
                        camera_name=args.camera_name,
                        camera_pos=pos,
                        camera_fovy=float(fovy),
                        render_width=args.render_width,
                        render_height=args.render_height,
                        rotate_180=args.rotate_180,
                    )
                    mae, mse, rmse = _metrics(small, target)
                    fine_results.append((mae, mse, rmse, pos, float(fovy), raw, small))
                    print("fine", pos, fovy, mae, rmse, flush=True)

    fine_results.sort(key=lambda x: x[0])
    best = fine_results[0]

    def save_result(prefix: str, result) -> None:
        mae, mse, rmse, pos, fovy, raw, small = result
        imageio.imwrite(out / f"{prefix}_raw_128.png", raw)
        imageio.imwrite(out / f"{prefix}_policy_64.png", small)
        compare = np.concatenate(
            [
                cv2.resize(small, (128, 128), interpolation=cv2.INTER_NEAREST),
                cv2.resize(target, (128, 128), interpolation=cv2.INTER_NEAREST),
            ],
            axis=1,
        )
        cv2.putText(
            compare,
            f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) fovy={fovy:.2f} mae={mae:.3f}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        imageio.imwrite(out / f"{prefix}_vs_dataset.png", compare)

    imageio.imwrite(out / "dataset_64.png", target)
    save_result("best_coarse", best_coarse)
    save_result("best_fine", best)

    top_tiles = []
    for rank, result in enumerate(fine_results[:12], start=1):
        mae, mse, rmse, pos, fovy, raw, small = result
        tile = np.concatenate(
            [
                cv2.resize(small, (128, 128), interpolation=cv2.INTER_NEAREST),
                cv2.resize(target, (128, 128), interpolation=cv2.INTER_NEAREST),
            ],
            axis=1,
        )
        cv2.putText(
            tile,
            f"#{rank} x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} f={fovy:.1f} mae={mae:.3f}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        top_tiles.append(tile)
    imageio.imwrite(out / "top12_fine_vs_dataset.png", np.concatenate(top_tiles, axis=0))

    with open(out / "results.txt", "w") as f:
        f.write("best_coarse\n")
        f.write(f"mae={best_coarse[0]:.6f} mse={best_coarse[1]:.6f} rmse={best_coarse[2]:.6f} pos={best_coarse[3]} fovy={best_coarse[4]:.3f}\n")
        f.write("best_fine\n")
        f.write(f"mae={best[0]:.6f} mse={best[1]:.6f} rmse={best[2]:.6f} pos={best[3]} fovy={best[4]:.3f}\n")

    print("BEST_COARSE", best_coarse[3], best_coarse[4], best_coarse[0], best_coarse[2])
    print("BEST_FINE", best[3], best[4], best[0], best[2])
    print("OUT", out)


if __name__ == "__main__":
    main()
