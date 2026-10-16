import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
os.makedirs(f"{R}/plots", exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))   # python's json reads the NaN/Infinity values in the logs
    return rows

def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)

def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if "step" in r and finite(r.get(key)):
            xs.append(r["step"]); ys.append(r[key])
    return xs, ys

def smooth(ys, w=10):
    out = []
    for i in range(len(ys)):
        chunk = ys[max(0, i - w + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out

def best_step(rows):
    steps = [r["step"] for r in rows if r.get("saved_best")]
    return steps[-1] if steps else None

# ---------- SFT ----------
sft = load_jsonl(f"{R}/sft/metrics.jsonl")
fig, ax = plt.subplots(figsize=(6.5, 4))
x, y = series(sft, "train_loss")
ax.plot(x, y, alpha=0.3, label="train loss (raw, logged every 10 steps)")
ax.plot(x, smooth(y, 10), label="train loss (moving average of 10 logged points)")
x, y = series(sft, "val_loss")
ax.plot(x, y, "o-", label="validation loss (1,000 examples, every 100 steps)")
b = best_step(sft)
if b is not None:
    ax.axvline(b, linestyle="--", color="gray", label=f"selected checkpoint (step {b})")
ax.set_xlabel("optimizer step"); ax.set_ylabel("token-mean NLL"); ax.set_title("SFT (Qwen2.5-0.5B, 30k examples)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{R}/plots/plot_sft.png", dpi=200); plt.close(fig)

# ---------- DPO (both runs) ----------
runs = [("HH-RLHF DPO", "dpo_hh"), ("UltraFeedback DPO", "dpo_uf")]
fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4 * len(runs)), squeeze=False)
for row, (label, folder) in enumerate(runs):
    d = load_jsonl(f"{R}/{folder}/metrics.jsonl")
    b = best_step(d)
    ax = axes[row][0]
    x, y = series(d, "train_loss")
    ax.plot(x, y, alpha=0.3, label="train loss (raw, every 5 steps)")
    ax.plot(x, smooth(y, 10), label="train loss (moving avg, 10 logged points)")
    x, y = series(d, "val_loss")
    ax.plot(x, y, "o-", label="validation loss")
    ax.set_ylabel("DPO loss")
    ax = axes[row][1]
    x, y = series(d, "train_reward_accuracy")
    ax.plot(x, smooth(y, 10), alpha=0.6, label="train reward accuracy (moving avg)")
    x, y = series(d, "val_reward_accuracy")
    ax.plot(x, y, "o-", label="validation reward accuracy")
    ax.set_ylabel("reward accuracy")
    ax = axes[row][2]
    x, y = series(d, "train_reward_margin")
    ax.plot(x, smooth(y, 10), alpha=0.6, label="train reward margin (moving avg)")
    x, y = series(d, "val_reward_margin")
    ax.plot(x, y, "o-", label="validation reward margin")
    ax.set_ylabel("reward margin")
    for ax in axes[row]:
        ax.set_xlabel("optimizer step"); ax.set_title(label)
        if b is not None:
            ax.axvline(b, linestyle="--", color="gray", label=f"selected checkpoint (step {b})")
        ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{R}/plots/plot_dpo.png", dpi=200); plt.close(fig)

# ---------- LoRA-DPO (three learning rates) ----------
lruns = [("lr 5e-5", "lora_persona"), ("lr 1e-4", "lora_persona_lr1e-4"), ("lr 2e-4", "lora_persona_lr2e-4")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, folder in lruns:
    h = json.load(open(f"{R}/{folder}/log_history.json"))
    tr = [e for e in h if "loss" in e and "eval_loss" not in e]
    ev = [e for e in h if "eval_loss" in e]
    xk = "step" if all("step" in e for e in tr + ev) else "epoch"
    line, = axes[0].plot([e[xk] for e in tr], [e["loss"] for e in tr], alpha=0.7, label=f"{label} train")
    axes[0].plot([e[xk] for e in ev], [e["eval_loss"] for e in ev], "o--", color=line.get_color(), label=f"{label} validation (end of each epoch)")
    axes[1].plot([e[xk] for e in tr], [e["rewards/margins"] for e in tr], color=line.get_color(), label=f"{label} train")
    axes[1].plot([e[xk] for e in ev], [e["eval_rewards/margins"] for e in ev], "o--", color=line.get_color(), label=f"{label} validation")
axes[0].set_yscale("log"); axes[0].set_ylabel("DPO loss (log scale)")
axes[1].set_ylabel("reward margin")
for ax in axes:
    ax.set_xlabel("optimizer step" if xk == "step" else "epoch"); ax.legend(fontsize=7)
axes[0].set_title("LoRA-DPO, SmolLM2-360M-Instruct: loss"); axes[1].set_title("LoRA-DPO: reward margin")
fig.tight_layout(); fig.savefig(f"{R}/plots/plot_lora.png", dpi=200); plt.close(fig)
print("saved plots in results/plots/")