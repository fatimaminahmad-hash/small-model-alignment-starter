import torch
import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class PadEqualsEosTokenizer(ToyTokenizer):
    def __init__(self):
        super().__init__()
        self.pad_token_id = self.eos_token_id  # like Qwen


def test_truncation_can_drop_eos_and_respects_max_length():
    tok = ToyTokenizer()
    # prompt = 6 tokens, response "ok" + EOS = 3 tokens -> 9 total; cut to 8
    batch = adapter.build_sft_batch(tok, ["Task: "], ["ok"], max_length=8)
    assert batch["input_ids"].shape[1] == 8
    assert batch["labels"][0, -1] != tok.eos_token_id
    assert batch["labels"][0, -1] != -100


def test_prompt_longer_than_max_length_gives_no_response_labels():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["Task: abcdef"], ["ok"], max_length=5)
    assert batch["labels"].eq(-100).all()


def test_mixed_lengths_keep_row_order_and_right_padding():
    tok = ToyTokenizer()
    prompts, responses = ["A: ", "Longer prompt: "], ["x", "yy"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=32)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        expected = tok.encode(p) + tok.encode(r) + [tok.eos_token_id]
        n = len(expected)
        assert batch["input_ids"][i, :n].tolist() == expected
        assert batch["attention_mask"][i, :n].eq(1).all()
        assert batch["attention_mask"][i, n:].eq(0).all()


def test_logprobs_do_not_depend_on_batch_neighbours():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.7)
    prompts, responses = ["Q: ", "A much longer prompt: "], ["hi", "there you go"]
    together = adapter.compute_response_logprobs(model, tok, prompts, responses, 64)
    alone = torch.stack([
        adapter.compute_response_logprobs(model, tok, [p], [r], 64)[0]
        for p, r in zip(prompts, responses)
    ])
    torch.testing.assert_close(together, alone)


def test_no_eos_tokenizer_adds_no_eos():
    tok = NoEosTokenizer()
    b = adapter.build_sft_batch(tok, ["Q: "], ["ok"], max_length=16)
    n_prompt = len(tok.encode("Q: "))
    assert b["labels"][0].ne(-100).sum() == len(tok.encode("ok"))
    assert b["input_ids"].shape[1] == n_prompt + 2


def test_pad_equals_eos_still_masks_correctly():
    tok = PadEqualsEosTokenizer()
    b = adapter.build_sft_batch(tok, ["Q: ", "Longer: "], ["a", "bbb"], max_length=32)
    assert b["labels"][0].ne(-100).sum() == 2
    assert b["labels"][0][b["attention_mask"][0].eq(0)].eq(-100).all()
    for i in range(2):
        last = b["attention_mask"][i].sum() - 1
        assert b["labels"][i, last] == tok.eos_token_id


def test_max_length_one_does_not_crash():
    tok = ToyTokenizer()
    b = adapter.build_sft_batch(tok, ["Q: "], ["ok"], max_length=1)
    assert b["input_ids"].shape == (1, 1)
