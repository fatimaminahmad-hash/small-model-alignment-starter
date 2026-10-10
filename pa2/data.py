"""Prompt formatting and SFT/DPO batch building."""
import torch

IGNORE_INDEX = -100

# One template, used for SFT, DPO and generation.
TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    return TEMPLATE.format(instruction=instruction.strip())


def _pad_id(tokenizer):
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return value
    return 0


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Right-padded batch; labels are -100 on prompt and padding tokens.

    EOS is appended to each response BEFORE truncation, so a long response can
    lose its EOS. Row order is preserved.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = _pad_id(tokenizer)

    rows = []
    for prompt, response in zip(prompts, responses):
        p = tokenizer.encode(prompt, add_special_tokens=False)
        r = tokenizer.encode(response, add_special_tokens=False)
        if eos is not None:
            r = r + [eos]
        ids = (p + r)[:max_length]
        labels = ([IGNORE_INDEX] * len(p) + r)[:max_length]
        rows.append((ids, labels))

    width = max((len(ids) for ids, _ in rows), default=0)
    input_ids, attention_mask, labels = [], [], []
    for ids, lab in rows:
        n_pad = width - len(ids)
        input_ids.append(ids + [pad] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels.append(lab + [IGNORE_INDEX] * n_pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Chosen and rejected get the identical prompt and the same masking rules."""
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }
