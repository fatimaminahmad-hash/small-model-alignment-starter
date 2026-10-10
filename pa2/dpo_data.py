"""Single-turn preference pairs for DPO: HH-RLHF and UltraFeedback."""
import random
from datasets import load_dataset

from pa2.data import format_prompt


def _split_hh(text):
    """'\\n\\nHuman: ...\\n\\nAssistant: ...' -> (prompt, response), or None if multi-turn."""
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    human, assistant = text.split("\n\nAssistant:")
    return human.split("\n\nHuman:")[1].strip(), assistant.strip()


def _hh_pairs(subset, split):
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    out = []
    for ex in ds:
        c, r = _split_hh(ex["chosen"]), _split_hh(ex["rejected"])
        if c is None or r is None or c[0] != r[0]:
            continue
        if not c[1] or not r[1] or c[1] == r[1]:
            continue
        out.append({"prompt": c[0], "chosen": c[1], "rejected": r[1], "source": subset})
    return out


def _uf_pairs(split):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    out = []
    for ex in ds:
        if len(ex["chosen"]) != 2 or len(ex["rejected"]) != 2:
            continue
        c, r = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
        if not c or not r or c == r:
            continue
        out.append({"prompt": ex["prompt"].strip(), "chosen": c, "rejected": r, "source": "ultrafeedback"})
    return out


def _take(pairs, n, tok, max_prompt_len, rng):
    rng.shuffle(pairs)
    out = []
    for p in pairs:
        formatted = format_prompt(p["prompt"])
        if len(tok.encode(formatted, add_special_tokens=False)) > max_prompt_len:
            continue
        out.append({**p, "prompt": formatted})
        if len(out) >= n:
            break
    return out


def load_dpo_data(name, n_train, n_val, tok, max_prompt_len=256, seed=42):
    rng = random.Random(seed)
    if name == "hh":
        half_t, half_v = n_train // 2, n_val // 2
        train = (_take(_hh_pairs("helpful-base", "train"), half_t, tok, max_prompt_len, rng)
                 + _take(_hh_pairs("harmless-base", "train"), n_train - half_t, tok, max_prompt_len, rng))
        val = (_take(_hh_pairs("helpful-base", "test"), half_v, tok, max_prompt_len, rng)
               + _take(_hh_pairs("harmless-base", "test"), n_val - half_v, tok, max_prompt_len, rng))
        rng.shuffle(train); rng.shuffle(val)
    elif name == "uf":
        train = _take(_uf_pairs("train_prefs"), n_train, tok, max_prompt_len, rng)
        val = _take(_uf_pairs("test_prefs"), n_val, tok, max_prompt_len, rng)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return train, val
