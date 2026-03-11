# Open-Model Bias-Rewrite Comparison (Qwen vs Llama)

This write-up compares:

- `llm_prompting_runner/outputs_qwen`
- `llm_prompting_runner/outputs_llama`

Both runs use the same evaluation size (`n = 18,150`) and the same prompting strategies:

- `zero_shot`
- `few_shot`
- `with_bias_tags`
- `npov`
- `self_refine`

## 1) New Metrics First: Bias Rewrite Effectiveness

### `BiasPhraseRetentionRate` (Primary)

Definition (lower is better): fraction of source biased phrases (BIO span-based) that still appear in prediction.

- `0.0` means all tagged biased phrases were removed/rewritten.
- `1.0` means all tagged biased phrases were retained.

This is the most direct metric for your objective ("is the model removing biased phrases out-of-box?").

### `BiasRetentionRate` (Secondary)

Definition (lower is better): density-style ratio based on retagging token counts in source vs prediction.

Use as supporting evidence only, because it depends more on tagger behavior and tokenization effects.

### `OverEditRate` (Control Metric)

Definition (lower is better): extra edit distance beyond reference-relative edit level.

Interpretation:

- low value -> safer, less distortion
- high value -> model is over-rewriting or drifting

## 2) Core Findings on Bias Rewrite

### Per-strategy `BiasPhraseRetentionRate`

| Strategy | Llama-3-8B | Qwen-2.5-7B | Better |
|---|---:|---:|---|
| zero_shot | 0.7052 | 0.8033 | Llama |
| few_shot | 0.8195 | 0.8999 | Llama |
| with_bias_tags | 0.7941 | 0.7817 | Qwen (slight) |
| npov | 0.7062 | 0.8028 | Llama |
| self_refine | 0.7404 | 0.8019 | Llama |

Summary:

- Llama is better on phrase-level bias removal in **4/5** strategies.
- Qwen only wins slightly on `with_bias_tags`.
- Best absolute bias-removal score across both models: **Llama zero-shot / npov (~0.705)**.

## 3) Trade-off: Bias Removal vs Rewrite Fidelity

### `few_shot` (strongest quality strategy for both)

- **Llama few_shot**
  - BLEU: `0.6225`
  - SemanticSimilarity: `0.9212`
  - BERTScoreF1: `0.7852`
  - METEOR: `0.7957`
  - BiasPhraseRetentionRate: `0.8195`
  - OverEditRate: `0.1413`
- **Qwen few_shot**
  - BLEU: `0.5855`
  - SemanticSimilarity: `0.9336`
  - BERTScoreF1: `0.7485`
  - METEOR: `0.8513`
  - BiasPhraseRetentionRate: `0.8999`
  - OverEditRate: `0.0885`

Interpretation:

- Llama few-shot rewrites more assertively (better bias removal, higher over-edit).
- Qwen few-shot is more conservative (better meaning preservation, weaker bias removal).

## 4) Strategy-by-Strategy Comparison

### zero_shot

- Llama clearly removes more bias (`0.7052` vs `0.8033`) but edits much more (`0.3199` vs `0.1831`).
- Qwen is safer but less effective for debiasing.

### with_bias_tags

- Qwen slightly better for phrase retention (`0.7817` vs `0.7941`) and lower over-edit.
- This is the only strategy where Qwen beats Llama on the primary debias metric.

### npov

- Similar pattern to zero-shot: Llama better debiasing, Qwen lower over-edit and better fidelity.

### self_refine

- Llama improves bias phrase retention versus Qwen (`0.7404` vs `0.8019`) but has heavy over-edit (`0.4872`).
- Qwen self-refine is less aggressive and more stable in fidelity metrics.

## 5) Qualitative Reliability Checks (Pipeline Behavior)

Counts are from output JSONL scans (all strategies, both models):

- assistant/meta prefaces: near zero in both runs
- `<bias>` leakage in prediction: zero in both runs
- pathological short/long outputs: low in both
- mojibake artifacts: non-zero in both, lower for Llama few-shot than Qwen few-shot

Notable difference:

- Qwen produces many more unchanged outputs in `few_shot` (`1870`) than Llama (`647`), consistent with lower over-edit and weaker bias removal.

## 6) Final Judgment for Your Goal

Your goal: evaluate whether these open models are good for **out-of-box bias rewrite**.

Answer from results:

1. Both models can produce usable rewrites.
2. Both still retain substantial bias phrases (all strategies remain far above ideal).
3. **Llama-3-8B is better for bias phrase removal overall.**
4. **Qwen-2.5-7B is better for conservative, high-fidelity rewriting.**

If your primary objective is debiasing effectiveness, pick:

- `meta-llama/llama-3-8b-instruct`, prefer `zero_shot` or `npov` for strongest bias removal signal, or `few_shot` for better text quality balance.

If your primary objective is minimal edit and meaning preservation, pick:

- `qwen/qwen-2.5-7b-instruct`, `few_shot`.

## 7) Recommendation for Reporting

Use:

- `BiasPhraseRetentionRate` as primary debias metric
- `BiasRetentionRate` as secondary
- `OverEditRate` to show rewrite-control tradeoff
- BLEU/BERTScore/Semantic/METEOR as quality context

This framing best matches your project question and avoids overclaiming from token-count-only bias metrics.

## 8) Baseline Comparison with T5 (From Project Report)

The project PDF (`DL_Grp_Project_Report.pdf`) reports the best T5 rewrite result (Table 3, 3 epochs, LR=0.0003):

- BLEU: `0.5068`
- Token-Level Accuracy: `0.3303`
- Semantic Similarity: `0.5803`

Compared with best open-model setting in this study (`Llama-3-8B`, `few_shot`):

- BLEU: `0.6225` (higher than T5 baseline)
- Token-Level Accuracy: `0.0386` (much lower than T5 baseline)
- Semantic Similarity: `0.9212` (higher than T5 baseline)

Compared with `Qwen-2.5-7B`, `few_shot`:

- BLEU: `0.5855` (higher than T5 baseline)
- Token-Level Accuracy: `0.0275` (much lower than T5 baseline)
- Semantic Similarity: `0.9336` (higher than T5 baseline)

Interpretation:

1. Out-of-box prompting models outperform the reported T5 baseline on BLEU and semantic similarity.
2. The T5 baseline strongly outperforms both open models on exact token-level match.
3. This pattern is consistent with model behavior:
   - T5 (task-trained) tends to stay closer to reference wording.
   - Prompted open models are more free-form paraphrasers.

Important caveat:

- This is a directional comparison, not a strict controlled benchmark, because the report's T5 training/evaluation setup differs from the out-of-box prompted LLM setup.
