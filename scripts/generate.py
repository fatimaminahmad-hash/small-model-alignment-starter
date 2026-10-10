import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pa2.data import format_prompt

p = argparse.ArgumentParser()
p.add_argument("--model_path", required=True)
p.add_argument("--tokenizer_path", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--generator", required=True)       # e.g. qwen2.5-0.5b-base, sft, dpo-hh
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--dataset", default="alpaca_eval")  # use hh_rlhf_style_eval for the HH set
p.add_argument("--max_new_tokens", type=int, default=256)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--repetition_penalty", type=float, default=1.0)
p.add_argument("--no_repeat_ngram_size", type=int, default=0)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer_path)
tok.padding_side = "left"
tok.truncation_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.float16).to("cuda:0").eval()
prompts = json.load(open(a.prompts))

results = []
for i in range(0, len(prompts), a.batch_size):
    chunk = prompts[i:i + a.batch_size]
    enc = tok([format_prompt(c["instruction"]) for c in chunk], return_tensors="pt",
              padding=True, truncation=True, max_length=512).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                             repetition_penalty=a.repetition_penalty,
                             no_repeat_ngram_size=a.no_repeat_ngram_size,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    for c, o in zip(chunk, out):
        text = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        rec = {k: v for k, v in c.items() if k != "dataset"}   # keeps id, instruction, category, ...
        rec.update({"output": text, "generator": a.generator, "dataset": a.dataset})
        results.append(rec)
    print(f"{len(results)}/{len(prompts)}", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(results, open(a.out, "w"), indent=2)
print("saved", a.out)
