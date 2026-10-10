"""DPO loss, implicit-reward metrics, and model-level DPO loss."""
import torch
import torch.nn.functional as F

from pa2.data import build_dpo_batch
from pa2.logprobs import sequence_logprobs


def dpo_terms(policy_chosen_logps, policy_rejected_logps,
              reference_chosen_logps, reference_rejected_logps, beta):
    """Per-example loss plus implicit rewards. Reference values are detached."""
    rc = reference_chosen_logps.detach()
    rr = reference_rejected_logps.detach()
    reward_chosen = beta * (policy_chosen_logps - rc)
    reward_rejected = beta * (policy_rejected_logps - rr)
    logits = reward_chosen - reward_rejected   # = beta * (delta_policy - delta_ref)
    losses = -F.logsigmoid(logits)             # stable form of -log(sigmoid(x))
    return losses, reward_chosen, reward_rejected


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    losses, _, _ = dpo_terms(policy_chosen_logps, policy_rejected_logps,
                             reference_chosen_logps, reference_rejected_logps, beta)
    return losses.mean()


@torch.no_grad()
def dpo_metrics(policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps, beta):
    """Reward accuracy uses strict r_w > r_l, so tied rewards count as wrong."""
    losses, rw, rl = dpo_terms(policy_chosen_logps, policy_rejected_logps,
                               reference_chosen_logps, reference_rejected_logps, beta)
    return {
        "loss": losses.mean().item(),
        "reward_accuracy": (rw > rl).float().mean().item(),
        "reward_margin": (rw - rl).mean().item(),
    }


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    pc, _ = sequence_logprobs(policy_model, batch["chosen"])
    pr, _ = sequence_logprobs(policy_model, batch["rejected"])
    with torch.no_grad():                       # the reference never gets gradients
        rc, _ = sequence_logprobs(reference_model, batch["chosen"])
        rr, _ = sequence_logprobs(reference_model, batch["rejected"])
    # the reference may live on another GPU
    rc, rr = rc.to(pc.device), rr.to(pc.device)
    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
