import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M-Instruct")
p.add_argument("--adapter", default=None)
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--generator", required=True)
p.add_argument("--dataset", default="persona_style_eval")
p.add_argument("--max_new_tokens", type=int, default=200)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--repetition_penalty", type=float, default=1.1)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32).to("cuda:0")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()

prompts = json.load(open(a.prompts))
results = []
for i in range(0, len(prompts), a.batch_size):
    chunk = prompts[i:i + a.batch_size]
    texts = [tok.apply_chat_template([{"role": "user", "content": c["instruction"]}],
                                     tokenize=False, add_generation_prompt=True) for c in chunk]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                             repetition_penalty=a.repetition_penalty, pad_token_id=tok.pad_token_id)
    for c, o in zip(chunk, out):
        rec = {k: v for k, v in c.items() if k != "dataset"}
        rec.update({"output": tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip(),
                    "generator": a.generator, "dataset": a.dataset})
        results.append(rec)
    print(f"{len(results)}/{len(prompts)}", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(results, open(a.out, "w"), indent=2)
print("saved", a.out)
