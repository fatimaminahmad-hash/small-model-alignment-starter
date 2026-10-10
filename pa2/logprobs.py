"""Response-only log-probabilities and the SFT loss."""
import torch

from pa2.data import IGNORE_INDEX, build_sft_batch


def _model_device(model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def sequence_logprobs(model, batch):
    """Return (summed response log-prob per row, response-token count per row)."""
    device = _model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()      # position t predicts token t+1
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.clamp_min(0)     # masked slots are zeroed out below

    token_logp = (
        logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        - torch.logsumexp(logits, dim=-1)
    )
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1), mask.sum(dim=-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, _ = sequence_logprobs(model, batch)
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Mean negative log-likelihood over all unmasked response tokens in the batch."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, counts = sequence_logprobs(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)
