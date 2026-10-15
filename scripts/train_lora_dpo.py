import argparse, json, os, random, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
p.add_argument("--out_dir", default="outputs/lora_persona")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--val_size", type=int, default=22)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

rows = json.load(open(a.data))
random.Random(a.seed).shuffle(rows)
val_rows, train_rows = rows[:a.val_size], rows[a.val_size:]

def conv(r):
    return {"prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}]}
train_ds = Dataset.from_list([conv(r) for r in train_rows])
val_ds = Dataset.from_list([conv(r) for r in val_rows])
print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)

lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg = DPOConfig(output_dir=a.out_dir, beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
                per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
                gradient_accumulation_steps=a.grad_accum, max_length=a.max_length,
                lr_scheduler_type="cosine", warmup_ratio=0.1, logging_steps=1,
                eval_strategy="epoch", save_strategy="no", report_to="none",
                seed=a.seed, fp16=False, bf16=False)
kwargs = dict(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, peft_config=lora)
try:
    trainer = DPOTrainer(processing_class=tok, **kwargs)
except TypeError:
    trainer = DPOTrainer(tokenizer=tok, **kwargs)
trainer.model.print_trainable_parameters()

trainer.train()
os.makedirs(a.out_dir, exist_ok=True)
trainer.save_model(os.path.join(a.out_dir, "adapter"))
json.dump(trainer.state.log_history, open(os.path.join(a.out_dir, "log_history.json"), "w"), indent=2)
json.dump({**vars(a), "n_train": len(train_rows), "n_val": len(val_rows)},
          open(os.path.join(a.out_dir, "args.json"), "w"), indent=2)
print("done")
