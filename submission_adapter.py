"""Stable grading interface for the SFT and DPO implementation.

Each function is a thin wrapper around the implementation in pa2/.
"""
from __future__ import annotations

from pa2 import data, dpo, logprobs


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""
    return data.build_sft_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""
    return logprobs.compute_response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss (token-mean reduction)."""
    return logprobs.compute_sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
):
    """Return separate response-masked LM batches for chosen and rejected rows."""
    return data.build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""
    return dpo.compute_dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps, beta)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""
    return dpo.compute_dpo_loss(
        policy_model, reference_model, tokenizer, prompts,
        chosen_responses, rejected_responses, beta, max_length)
