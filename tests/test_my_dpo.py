import math
import torch
import submission_adapter as adapter
from pa2.dpo import dpo_metrics
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_known_value():
    # beta=0.1, delta_policy=2, delta_ref=0 -> loss = softplus(-0.2)
    pc, pr = torch.tensor([-1.0]), torch.tensor([-3.0])
    rc, rr = torch.tensor([-2.0]), torch.tensor([-2.0])
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta=0.1)
    assert abs(loss.item() - math.log(1 + math.exp(-0.2))) < 1e-6


def test_loss_drops_when_policy_prefers_chosen_more():
    rc = rr = torch.tensor([-2.0, -2.0])
    base = adapter.compute_dpo_loss_from_logps(torch.tensor([-2.0, -2.0]), torch.tensor([-2.0, -2.0]), rc, rr, 0.1)
    better = adapter.compute_dpo_loss_from_logps(torch.tensor([-1.0, -1.0]), torch.tensor([-3.0, -3.0]), rc, rr, 0.1)
    assert better < base


def test_reference_logps_get_no_gradient():
    pc = torch.tensor([-1.0, -2.0], requires_grad=True)
    pr = torch.tensor([-3.0, -2.5], requires_grad=True)
    rc = torch.tensor([-2.0, -2.0], requires_grad=True)
    rr = torch.tensor([-2.0, -2.0], requires_grad=True)
    adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.1).backward()
    assert pc.grad is not None and rc.grad is None and rr.grad is None


def test_reference_model_untouched_policy_gets_grads():
    tok = ToyTokenizer()
    policy, reference = ToyCausalLM(tok.vocab_size), ToyCausalLM(tok.vocab_size)
    before = reference.logit_bias.detach().clone()
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["Pick: ", "Go: "], ["a", "a"], ["b", "b"], 0.1, 16)
    loss.backward()
    assert reference.logit_bias.grad is None
    assert torch.equal(reference.logit_bias.detach(), before)
    assert policy.logit_bias.grad.abs().sum() > 0


def test_ties_do_not_count_as_correct():
    v = torch.tensor([-2.0, -5.0])
    m = dpo_metrics(v, v, v, v, 0.1)
    assert m["reward_accuracy"] == 0.0 and m["reward_margin"] == 0.0
