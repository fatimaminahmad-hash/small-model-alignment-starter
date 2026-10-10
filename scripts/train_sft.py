import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2.data import build_sft_batch
from pa2.logprobs import sequence_logprobs
from pa2.sft_data import load_ultrachat_single_turn


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out_dir", default="outputs/sft")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, tok, data, bs, max_length):
    """Token-mean validation NLL over the whole validation set."""
    model.eval()
    nll, ntok = 0.0, 0
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        batch = build_sft_batch(tok, [d["prompt"] for d in chunk],
                                [d["response"] for d in chunk], max_length)
        with torch.autocast("cuda", dtype=torch.float16):
            logps, counts = sequence_logprobs(model, batch)
        nll += -logps.sum().item()
        ntok += counts.sum().item()
    model.train()
    return nll / max(ntok, 1)


def save(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def main():
    a = parse()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out_dir, "args.json"), "w"), indent=2)
    metrics_path = os.path.join(a.out_dir, "metrics.jsonl")

    train, val = load_ultrachat_single_turn(a.n_train, a.n_val, seed=a.seed)
    print(f"train={len(train)} val={len(val)}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to("cuda:0")
    model.config.use_cache = False
    model.train()

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)

    per_step = a.batch_size * a.grad_accum
    steps_per_epoch = len(train) // per_step
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, int(a.warmup_ratio * total_steps))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    rng = random.Random(a.seed)
    order = []
    for _ in range(a.epochs):
        idx = list(range(len(train))); rng.shuffle(idx); order += idx

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    best = float("inf")
    v0 = evaluate(model, tok, val, a.eval_batch_size, a.max_length)
    log({"step": 0, "val_loss": v0})
    t0 = time.time()

    for step in range(total_steps):
        chunk = order[step * per_step:(step + 1) * per_step]
        step_loss = 0.0
        for m in range(a.grad_accum):
            mb = [train[j] for j in chunk[m * a.batch_size:(m + 1) * a.batch_size]]
            batch = build_sft_batch(tok, [d["prompt"] for d in mb],
                                    [d["response"] for d in mb], a.max_length)
            with torch.autocast("cuda", dtype=torch.float16):
                logps, counts = sequence_logprobs(model, batch)
            loss = -logps.sum() / counts.sum().clamp_min(1)
            scaler.scale(loss / a.grad_accum).backward()
            step_loss += loss.item() / a.grad_accum

        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True); sched.step()
        s = step + 1

        if s % a.log_every == 0 or s == 1:
            log({"step": s, "train_loss": step_loss, "grad_norm": float(gn),
                 "lr": sched.get_last_lr()[0], "elapsed_min": (time.time() - t0) / 60})
        if s % a.eval_every == 0 or s == total_steps:
            v = evaluate(model, tok, val, a.eval_batch_size, a.max_length)
            log({"step": s, "val_loss": v})
            if v < best:
                best = v
                save(model, tok, os.path.join(a.out_dir, "best"))
                log({"step": s, "saved_best": True, "best_val_loss": best})

    save(model, tok, os.path.join(a.out_dir, "final"))
    log({"done": True, "best_val_loss": best})


if __name__ == "__main__":
    main()
