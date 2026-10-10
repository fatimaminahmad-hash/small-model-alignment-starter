import argparse, json, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pa2.data import build_dpo_batch
from pa2.dpo import dpo_metrics, dpo_terms
from pa2.dpo_data import load_dpo_data
from pa2.logprobs import sequence_logprobs


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hh", "uf"], required=True)
    p.add_argument("--sft_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=15000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pair_logps(model, tok, ex, max_length):
    batch = build_dpo_batch(tok, [ex["prompt"]], [ex["chosen"]], [ex["rejected"]], max_length)
    with torch.autocast("cuda", dtype=torch.float16):
        c, _ = sequence_logprobs(model, batch["chosen"])
        r, _ = sequence_logprobs(model, batch["rejected"])
    return c, r


@torch.no_grad()
def evaluate(policy, reference, tok, data, beta, max_length):
    policy.eval()
    pcs, prs, rcs, rrs = [], [], [], []
    for ex in data:
        pc, pr = pair_logps(policy, tok, ex, max_length)
        rc, rr = pair_logps(reference, tok, ex, max_length)
        pcs.append(pc.cpu()); prs.append(pr.cpu()); rcs.append(rc.cpu()); rrs.append(rr.cpu())
    policy.train()
    cat = lambda xs: torch.cat(xs)
    return dpo_metrics(cat(pcs), cat(prs), cat(rcs), cat(rrs), beta)


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

    tok = AutoTokenizer.from_pretrained(a.sft_path)
    train, val = load_dpo_data(a.dataset, a.n_train, a.n_val, tok, a.max_prompt_length, a.seed)
    print(f"dataset={a.dataset} train={len(train)} val={len(val)}", flush=True)

    policy = AutoModelForCausalLM.from_pretrained(a.sft_path, torch_dtype=torch.float32).to("cuda:0")
    policy.config.use_cache = False
    policy.train()
    reference = AutoModelForCausalLM.from_pretrained(a.sft_path, torch_dtype=torch.float16).to("cuda:1")
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    total_steps = len(train) // a.grad_accum
    rng = random.Random(a.seed)
    order = list(range(len(train))); rng.shuffle(order)

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    m0 = evaluate(policy, reference, tok, val, a.beta, a.max_length)
    log({"step": 0, **{f"val_{k}": v for k, v in m0.items()}})   # expect loss ~0.693, acc 0.0, margin 0.0

    best_acc, t0 = -1.0, time.time()
    for step in range(total_steps):
        chunk = order[step * a.grad_accum:(step + 1) * a.grad_accum]
        sl, sacc, smar = 0.0, 0.0, 0.0
        for j in chunk:
            ex = train[j]
            pc, pr = pair_logps(policy, tok, ex, a.max_length)
            with torch.no_grad():
                rc, rr = pair_logps(reference, tok, ex, a.max_length)
            losses, rw, rl = dpo_terms(pc, pr, rc.to(pc.device), rr.to(pc.device), a.beta)
            loss = losses.mean()
            scaler.scale(loss / a.grad_accum).backward()
            sl += loss.item() / a.grad_accum
            sacc += float((rw > rl).item()) / a.grad_accum
            smar += (rw - rl).mean().item() / a.grad_accum

        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), a.clip)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        s = step + 1

        if s % a.log_every == 0 or s == 1:
            log({"step": s, "train_loss": sl, "train_reward_accuracy": sacc,
                 "train_reward_margin": smar, "grad_norm": float(gn),
                 "elapsed_min": (time.time() - t0) / 60})
        if s % a.eval_every == 0 or s == total_steps:
            m = evaluate(policy, reference, tok, val, a.beta, a.max_length)
            log({"step": s, **{f"val_{k}": v for k, v in m.items()}})
            if m["reward_accuracy"] > best_acc:
                best_acc = m["reward_accuracy"]
                save(policy, tok, os.path.join(a.out_dir, "best"))
                log({"step": s, "saved_best": True, "best_val_reward_accuracy": best_acc})

    save(policy, tok, os.path.join(a.out_dir, "final"))
    log({"done": True, "best_val_reward_accuracy": best_acc})


if __name__ == "__main__":
    main()
