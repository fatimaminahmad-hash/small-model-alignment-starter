"""Single-turn UltraChat data for SFT."""
from datasets import load_dataset

from pa2.data import format_prompt


def load_ultrachat_single_turn(n_train, n_val, seed=42,
                               dataset="HuggingFaceH4/ultrachat_200k",
                               split="train_sft", max_instruction_chars=1500):
    ds = load_dataset(dataset, split=split).shuffle(seed=seed)
    need = n_train + n_val
    out = []
    for ex in ds:
        msgs = ex["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        instr, resp = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not instr or not resp or len(instr) > max_instruction_chars:
            continue
        out.append({"prompt": format_prompt(instr), "response": resp})
        if len(out) >= need:
            break
    return out[:n_train], out[n_train:n_train + n_val]
