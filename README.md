# Small-Model Alignment: SFT + DPO (PA2)

Course project: supervised fine-tuning of `Qwen/Qwen2.5-0.5B`, two from-scratch DPO runs (HH-RLHF and UltraFeedback), a TRL `DPOTrainer` + LoRA persona run on `HuggingFaceTB/SmolLM2-360M-Instruct`, and pairwise LLM-judge evaluation (`google/gemini-3.6-flash` via the Kaggle model proxy).

The final report (PDF) is submitted separately. This README lets staff reproduce every number in it.

## Headline results (judge: `google/gemini-3.6-flash`, 0 judging failures in every file)

| Comparison (candidate vs baseline) | Eval set / rubric | n | W / T / L | Win rate (ties excluded) |
|---|---|---|---|---|
| DPO-HH vs SFT | HH-300 / safety rubric | 300 | 183 / 27 / 90 | 0.670 |
| DPO-UF vs SFT | AlpacaEval-300 / general rubric | 300 | 90 / 72 / 138 | 0.395 |
| LoRA persona (lr 2e-4) vs base | HH-300 / safety rubric | 300 | 69 / 14 / 217 | 0.241 |

Persona style / usefulness / safety / overall results (24 prompts) are reported separately in the report and in `results/judged/persona_*`. Different evaluation sets are never combined into one score.

## Repository layout

```text
submission_adapter.py        # graded entry point (six functions, thin wrappers over pa2/)
pa2/
  data.py                    # prompt template, SFT/DPO batch building (response-only masking)
  logprobs.py                # sequence log-probs, response-only; SFT loss (token mean)
  dpo.py                     # DPO loss, reward accuracy (strict >), reward margin
  sft_data.py                # UltraChat loading/filtering (30k train / 1k val)
  dpo_data.py                # HH-RLHF and UltraFeedback pair loading
scripts/
  train_sft.py               # SFT, own training loop
  train_dpo.py               # from-scratch DPO (--dataset hh|uf)
  make_alpaca_subset.py      # fixed 300-prompt AlpacaEval subset (seed 42)
  generate.py                # generation for Qwen models
  build_pairs.py             # randomized A/B pair files with hidden model mapping
  judge_pairwise_kaggle.py   # pairwise judge (Kaggle model proxy), resumable
  train_lora_dpo.py          # TRL DPOTrainer + PEFT LoRA (persona)
  generate_persona.py        # generation for SmolLM2 (+ optional LoRA adapter)
  judge_persona_kaggle.py    # four-verdict persona judge (style/useful/safe/overall)
  plot_metrics.py            # training plots
eval_sets/
  alpaca_eval_300_seed42.json
  hh_rlhf_eval_300_seed42.json        # provided
  persona_style_eval_24.json
data/persona_lora/ramsay_style_preferences.json   # provided persona preferences
tests/                       # 7 public tests + 12 of my own (19 total)
results/                     # all logs, generations, judged outputs, plots (see below)
```

## Environment

All experiments ran on Kaggle, GPU `T4 x2` (2 x Tesla T4, 15 GiB each).

| Package | Version used |
|---|---|
| torch | 2.11.0+cu128 (preinstalled on Kaggle; not pinned in `requirements.txt`) |
| transformers | 4.57.6 |
| datasets | 4.8.5 |
| trl | 1.15.0 |
| peft | 0.20.0 |

Setup (Kaggle or any CUDA machine with PyTorch already installed):

```bash
git clone https://github.com/fatimaminahmad-hash/small-model-alignment-starter.git
cd small-model-alignment-starter
python -m pip uninstall -y -q torchao || true
python -m pip install -q -r requirements.txt
python -m pip install -q trl
python -m pip install -q -e . --no-deps
```

Run the tests (no GPU, no downloads):

```bash
PYTHONPATH=. python -m pytest -q tests/
```

Expected: 19 passed (7 public adapter tests including the signature check, 12 additional tests).

## Fixed settings

- **Seed:** 42 everywhere (data subsets, shuffles, pair randomization, training, evaluation sampling).
- **Prompt template** (all Qwen runs and generation; plain text, no chat template):
  `### Instruction:\n{instruction}\n\n### Response:\n`
  Labels are `-100` on prompt and padding; EOS is appended before truncation; right padding. SmolLM2 uses its own chat template (TRL for training, `apply_chat_template` for generation).
- **SFT loss:** mean over unmasked response tokens (token mean).
- **Decoding for all Qwen comparisons:** greedy (`do_sample=False`), `max_new_tokens=256`, `repetition_penalty=1.1`, same on both sides of every comparison (`results/gen/decoding.txt`). Plain-greedy outputs are kept in `results/gen_greedy/`. Persona generation: greedy, `max_new_tokens=200`, `repetition_penalty=1.1`.
- **Data:**
  - SFT: single-turn UltraChat, instruction <= 1500 characters, 30,000 train / 1,000 val.
  - DPO-HH: `Anthropic/hh-rlhf` helpful-base + harmless-base, single-turn only (about 29% of conversations), identical prompt on both sides, distinct responses, prompt <= 256 tokens; 7,500 train pairs per subset and 500 val pairs per subset from the test splits (15,000 / 1,000).
  - DPO-UF: `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs` / `test_prefs`), 15,000 / 1,000.
  - LoRA: provided `ramsay_style_preferences.json`, 198 train / 22 val.

## Reproducing the pipeline

All commands are run from the repository root. Paths below are the ones used for the reported runs.

### 1. SFT (about 96 minutes on T4 x2)

```bash
python scripts/train_sft.py --out_dir outputs/sft
```

Defaults are the values used: batch 2, accumulation 16 (effective 32), lr 2e-5, cosine, warmup 0.03, weight decay 0.1, clip 1.0, 1 epoch (937 steps), max length 512, validation every 100 steps. Saves `outputs/sft/best` and `outputs/sft/final`. Selected: step 937 (best = final).

### 2. DPO from scratch (two runs, same SFT checkpoint as initial policy and frozen reference)

```bash
python scripts/train_dpo.py --dataset hh --sft_path outputs/sft/best --out_dir outputs/dpo_hh
python scripts/train_dpo.py --dataset uf --sft_path outputs/sft/best --out_dir outputs/dpo_uf
```

Defaults are the values used: beta 0.1, lr 1e-6 constant, per-device batch 1, accumulation 32 (effective 32), clip 1.0, max length 512 (prompt 256), 15,000 train / 1,000 val pairs, 1 epoch (468 steps), validation every 100 steps. Policy in fp32 on `cuda:0` (fp16 autocast), frozen fp16 reference on `cuda:1`. The checkpoint with the best validation reward accuracy is saved as `best` (HH: step 468; UF: step 400) and the last as `final`.

### 3. Baseline generation and fixed evaluation subsets

```bash
python scripts/make_alpaca_subset.py                       # writes eval_sets/alpaca_eval_300_seed42.json
```

Base, SFT and DPO generations (identical decoding on both sides of every comparison):

```bash
# AlpacaEval-300
python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \
  --prompts eval_sets/alpaca_eval_300_seed42.json --out results/gen/alpaca_base.json \
  --repetition_penalty 1.1 --max_new_tokens 256
python scripts/generate.py --model_path outputs/sft/best --generator sft \
  --prompts eval_sets/alpaca_eval_300_seed42.json --out results/gen/alpaca_sft.json \
  --repetition_penalty 1.1 --max_new_tokens 256
python scripts/generate.py --model_path outputs/dpo_uf/best --generator dpo-uf \
  --prompts eval_sets/alpaca_eval_300_seed42.json --out results/gen/alpaca_dpo_uf.json \
  --repetition_penalty 1.1 --max_new_tokens 256

# HH-300 (provided synthetic helpful/harmless set)
python scripts/generate.py --model_path outputs/sft/best --generator sft \
  --prompts eval_sets/hh_rlhf_eval_300_seed42.json --dataset hh_rlhf_style_eval \
  --out results/gen/hh_sft.json --repetition_penalty 1.1 --max_new_tokens 256
python scripts/generate.py --model_path outputs/dpo_hh/best --generator dpo-hh \
  --prompts eval_sets/hh_rlhf_eval_300_seed42.json --dataset hh_rlhf_style_eval \
  --out results/gen/hh_dpo.json --repetition_penalty 1.1 --max_new_tokens 256
```

Plain-greedy versions (no repetition penalty) in `results/gen_greedy/` use the same commands with `--repetition_penalty 1.0` and are not used for the reported comparisons.

### 4. Pairwise comparison files (randomized A/B, hidden mapping)

```bash
python scripts/build_pairs.py --dpo results/gen/hh_dpo.json --sft results/gen/hh_sft.json \
  --dpo_name dpo --sft_name sft --out results/pairs/hh_dpo_vs_sft_pairs.jsonl
python scripts/build_pairs.py --dpo results/gen/alpaca_dpo_uf.json --sft results/gen/alpaca_sft.json \
  --dpo_name dpo --sft_name sft --out results/pairs/uf_dpo_vs_sft_pairs.jsonl
```

Response order is shuffled with seed 42; the A/B-to-model mapping is stored only in the `hidden` field of each record and is never shown to the judge. The judge prompt labels outputs only as "Response A" and "Response B".

### 5. Judging (Kaggle benchmark notebook with `kaggle_benchmarks`)

The judge needs the Kaggle model proxy, so run this inside a Kaggle notebook configured as a utility benchmark.

```bash
# HH-300, safety rubric
python scripts/judge_pairwise_kaggle.py \
  --pairs_path results/pairs/hh_dpo_vs_sft_pairs.jsonl \
  --output_path results/judged/hh_dpo_vs_sft_judged.jsonl \
  --failures_path results/judged/hh_dpo_vs_sft_judged_failures.jsonl \
  --judge_model google/gemini-3.6-flash --safety --target_name dpo

# AlpacaEval-300, general rubric (run in chunks of 100 pairs to survive session loss)
python scripts/judge_pairwise_kaggle.py \
  --pairs_path results/pairs/uf_dpo_vs_sft_pairs.jsonl \
  --output_path results/judged/uf_dpo_vs_sft_judged.jsonl \
  --failures_path results/judged/uf_dpo_vs_sft_judged_failures.jsonl \
  --judge_model google/gemini-3.6-flash --target_name dpo
```

Judging protocol: batches of 5 items, at least 5 s sleep between calls, up to 3 retries per batch, resumable JSONL output (`--reuse_judgments`), unresolved items written to the `*_failures.jsonl` file (all empty in the reported runs: 0 failures). Win rate is reported both with ties excluded, W/(W+L), and with ties counted as half, (W+T/2)/n. The `--fake` flag runs the plumbing without model calls.

The UF comparison was judged twice because an interactive session was lost: first pass 93 / 77 / 130 (win rate 0.417), rerun 90 / 72 / 138 (0.395). The rerun is the reported result. Chunk files `uf_part1..3.jsonl` and `hh_lora_lr2e-4_vs_base_part1..3.jsonl` are the per-100-pair pieces merged into the final judged files.

### 6. LoRA-DPO persona track (TRL `DPOTrainer` + PEFT)

TRL requires a single visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
for LR in 5e-5 1e-4 2e-4; do
  python scripts/train_lora_dpo.py --lr $LR --out_dir outputs/lora_persona_lr$LR
done
```

Defaults are the values used: rank 16, alpha 32, dropout 0.05, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (8.68M of 370.5M trainable, 2.34%), batch 4, accumulation 4 (effective 16), 3 epochs (39 steps), beta 0.1, cosine schedule with warmup ratio 0.1, max length 512, 198 train / 22 val (seed 42), fp32. The reference is the same model with the adapter disabled. Adapters are saved to `outputs/lora_persona*/adapter` (not committed because of size). Logs: `results/lora_persona*/log_history.json`, `args.json`. (The directory without a suffix is the 5e-5 run.)

Persona generations (greedy, 200 new tokens, repetition penalty 1.1, chat template):

```bash
python scripts/generate_persona.py --generator base \
  --prompts eval_sets/persona_style_eval_24.json --out results/gen_persona/persona_base.json
python scripts/generate_persona.py --generator lora_lr1e-4 --adapter outputs/lora_persona_lr1e-4/adapter \
  --prompts eval_sets/persona_style_eval_24.json --out results/gen_persona/persona_lora_lr1e-4.json
# likewise for 5e-5 (persona_lora.json) and 2e-4 (persona_lora_lr2e-4.json)
```

Pair files and judging use the same pairing script and the persona judge (four verdicts per prompt: style, usefulness, safety, overall):

```bash
python scripts/build_pairs.py --dpo results/gen_persona/persona_lora_lr1e-4.json \
  --sft results/gen_persona/persona_base.json --dpo_name lora_lr1e-4 --sft_name base \
  --out results/pairs/persona_lr1e-4_vs_base_pairs.jsonl
python scripts/judge_persona_kaggle.py \
  --pairs_path results/pairs/persona_lr1e-4_vs_base_pairs.jsonl \
  --output_path results/judged/persona_lr1e-4_vs_base_judged.jsonl \
  --failures_path results/judged/persona_lr1e-4_vs_base_judged_failures.jsonl
```

The LoRA adapter (lr 2e-4) was also judged against its base on HH-300 with the safety rubric (`results/judged/hh_lora_lr2e-4_vs_base_judged.jsonl`).

### 7. Plots and analysis

```bash
python scripts/plot_metrics.py        # writes results/plots/plot_sft.png, plot_dpo.png, plot_lora.png
```

The UF length analysis is in `results/analysis/uf_length_split.txt` (response length = whitespace-split word count).

## Where each reported number lives

| What | Path |
|---|---|
| Hyperparameters actually used | `results/sft/args.json`, `results/dpo_hh/args.json`, `results/dpo_uf/args.json`, `results/lora_persona*/args.json` |
| SFT / DPO validation and training metrics | `results/sft/metrics.jsonl`, `results/dpo_hh/metrics.jsonl`, `results/dpo_uf/metrics.jsonl` |
| LoRA training logs | `results/lora_persona*/log_history.json` |
| Fixed evaluation sets | `eval_sets/` |
| Generations (final decoding) | `results/gen/`, `results/gen_persona/` |
| Generations (plain greedy) | `results/gen_greedy/` |
| Decoding settings | `results/gen/decoding.txt` |
| Pair files (hidden mapping) | `results/pairs/` |
| Judged outputs and failure logs | `results/judged/` |
| UF length split | `results/analysis/uf_length_split.txt` |
| Training plots | `results/plots/` |

Model weights (SFT, DPO checkpoints of about 2 GB each) and LoRA adapters are not committed because of size; the commands above regenerate them.

## Submission adapter

`submission_adapter.py` exposes the six required functions with unchanged signatures and delegates to `pa2/`. The SFT loss uses the mean over all unmasked response tokens. DPO reward accuracy uses strict `r_w > r_l`, so it is exactly 0 when the policy equals the reference.

## Notes and known limitations

- One judge model (`google/gemini-3.6-flash`), one seed, no human validation of the judge.
- Small evaluation sets (300 prompts; 30 per HH category; 24 persona prompts), so only large effects are resolved.
- DPO used one fixed learning rate (1e-6) and one epoch; the HH run was still improving at its last step.
- Decoding uses a repetition penalty of 1.1 for all Qwen comparisons because plain greedy decoding produced repetition loops in roughly 35-40% of outputs (documented deviation; see the report).
- See the final report for the full configuration table, deviations, and error analysis.
