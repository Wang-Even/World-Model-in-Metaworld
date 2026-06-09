#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

ENV_NAME="${ENV_NAME:-button-press-v3}"
EPISODES="${EPISODES:-50}"
ROLLOUT_LEN="${ROLLOUT_LEN:-500}"
CONTEXT_LEN="${CONTEXT_LEN:-8}"
SEED="${SEED:-0}"
RENDER_WIDTH="${RENDER_WIDTH:-128}"
RENDER_HEIGHT="${RENDER_HEIGHT:-128}"
ENV_BACKEND="${ENV_BACKEND:-local}"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-.conda-veorl/bin/python}"
BRIDGE_SERVER="${BRIDGE_SERVER:-scripts/veorl_metaworld_server.py}"
BC_REW_CKPT="${BC_REW_CKPT:-logs/bc_rew_button_press/checkpoints}"
RL_CKPT="${RL_CKPT:-}"
SAVE_PREVIEW_EPISODES="${SAVE_PREVIEW_EPISODES:-1}"
SAVE_PREVIEW_FRAMES="${SAVE_PREVIEW_FRAMES:-8}"

CMD=(
  python scripts/metaworld-dreamer-rl.py
  --env "$ENV_NAME"
  --episodes "$EPISODES"
  --rollout_len "$ROLLOUT_LEN"
  --context_len "$CONTEXT_LEN"
  --seed "$SEED"
  --camera_name corner2
  --camera_pos 0.91 0.03 0.73
  --rotate_180
  --render_width "$RENDER_WIDTH"
  --render_height "$RENDER_HEIGHT"
  --env_backend "$ENV_BACKEND"
  --bridge_python "$BRIDGE_PYTHON"
  --bridge_server "$BRIDGE_SERVER"
  --save_preview_episodes "$SAVE_PREVIEW_EPISODES"
  --save_preview_frames "$SAVE_PREVIEW_FRAMES"
)

if [[ -n "$RL_CKPT" ]]; then
  CMD+=(--rl_ckpt "$RL_CKPT")
else
  CMD+=(--use_bc_head --bc_rew_ckpt "$BC_REW_CKPT")
fi

CMD+=("$@")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
