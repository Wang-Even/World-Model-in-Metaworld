from pathlib import Path
import orbax.checkpoint as ocp

# 换成你现在用的 dynamics ckpt 目录
ckpt_root = Path("logs/dynamics_button_press/checkpoints").resolve()

# 找所有 step 子目录（名字是数字）
steps = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
if not steps:
    print("该目录下没有 ckpt 子目录")
    raise SystemExit

latest = steps[-1]
state_dir = (ckpt_root / str(latest) / "state").resolve()
print("读取 ckpt 目录:", state_dir)

checkpointer = ocp.PyTreeCheckpointer()
state = checkpointer.restore(str(state_dir))

print("state 顶层 keys:", state.keys())

params = state["params"]
print("params 顶层 keys:", params.keys())

# 有的版本是 {'dyn': {...}}，有的直接就是 dyn params
dyn_params = params["dyn"] if "dyn" in params else params
print("dyn_params 顶层 keys:", dyn_params.keys())

if "action_encoder" in dyn_params:
    print("action_encoder 子树 keys:", dyn_params["action_encoder"].keys())
else:
    print("dyn_params 里面没有 action_encoder 这个子树")

