import json, random, os
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json", repo_type="dataset")
data = json.load(open(path))
print("total prompts:", len(data))

rng = random.Random(42)
idx = sorted(rng.sample(range(len(data)), 300))
subset = [{"id": f"alpaca_{i}", "instruction": data[i]["instruction"], "dataset": "alpaca_eval"} for i in idx]

os.makedirs("eval_sets", exist_ok=True)
json.dump(subset, open("eval_sets/alpaca_eval_300_seed42.json", "w"), indent=2)
print("saved", len(subset), "prompts")
