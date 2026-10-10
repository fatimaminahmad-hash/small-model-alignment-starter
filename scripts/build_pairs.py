import argparse, json, random

p = argparse.ArgumentParser()
p.add_argument("--dpo", required=True)
p.add_argument("--sft", required=True)
p.add_argument("--dpo_name", default="dpo")
p.add_argument("--sft_name", default="sft")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

dpo = {r["id"]: r for r in json.load(open(a.dpo))}
sft = {r["id"]: r for r in json.load(open(a.sft))}
ids = [i for i in dpo if i in sft]
rng = random.Random(a.seed)

n_a_is_dpo = 0
with open(a.out, "w") as f:
    for i in ids:
        d, s = dpo[i], sft[i]
        swap = rng.random() < 0.5
        first, second = (s, d) if swap else (d, s)
        a_is, b_is = (a.sft_name, a.dpo_name) if swap else (a.dpo_name, a.sft_name)
        n_a_is_dpo += (a_is == a.dpo_name)
        rec = {"id": i, "instruction": d["instruction"],
               "response_a": first["output"], "response_b": second["output"],
               "hidden": {"a_is": a_is, "b_is": b_is}}
        if d.get("category") is not None:
            rec["category"] = d["category"]
        f.write(json.dumps(rec) + "\n")
print(f"wrote {len(ids)} pairs to {a.out}; DPO is Response A in {n_a_is_dpo}")
