# Local Open-Weight Runner

This folder is the new project-direction workspace for prompting experiments, using OpenAI SDK against OpenAI-compatible endpoints (for example OpenRouter).

## What this runs

- Models:
  - Current default: `meta-llama/llama-3.1-8b-instruct`
  - Common tested models: `qwen/qwen-2.5-7b-instruct`, `meta-llama/llama-3-8b-instruct`
- Prompting strategies:
  - `zero_shot`
  - `few_shot` (3 examples by default)
  - `with_bias_tags` (uses existing BiLSTM tagger)
  - `npov`
  - `self_refine` (generate -> critique -> improve)
- Metrics:
  - BLEU
  - Token-Level Accuracy
  - BERTScore F1
  - Semantic Similarity
  - METEOR
  - Bias Retention Rate (retagging density; secondary)
  - Bias Phrase Retention Rate (phrase-level retention; primary)
  - Over-Edit Rate
  - No-Bias-Reduction Count (objective indicator)

## Setup (uv)

From repo root:

```powershell
uv venv
.venv\Scripts\activate
uv pip install -r llm_prompting_runner/requirements.txt
```

If METEOR is used, download NLTK data once:

```powershell
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Run

```powershell
uv run python llm_prompting_runner/run_local_models.py `
  --api_key_file llm_prompting_runner/.local.env `
  --api_base https://openrouter.ai/api/v1 `
  --models meta-llama/llama-3-8b-instruct
```

Optional example with smaller pilot:

```powershell
uv run python llm_prompting_runner/run_local_models.py `
  --samples 200 `
  --models qwen/qwen-2.5-7b-instruct meta-llama/llama-3-8b-instruct `
  --api_base https://openrouter.ai/api/v1
```

Full run example (all 18,150 samples, all strategies):

```powershell
uv run python llm_prompting_runner/run_local_models.py `
  --api_key_file llm_prompting_runner/.local.env `
  --api_base https://openrouter.ai/api/v1 `
  --models meta-llama/llama-3-8b-instruct `
  --strategies zero_shot few_shot with_bias_tags npov self_refine `
  --samples 18150 `
  --parallel_requests 10 `
  --request_timeout 90 `
  --max_tokens 180 `
  --quality_retry_attempts 1 `
  --output_dir llm_prompting_runner/outputs
```

After generation, rescore in place:

```powershell
uv run python llm_prompting_runner/rescore_outputs_inplace.py `
  --input_dir llm_prompting_runner/outputs
```

## CLI Args (Runner)

`run_local_models.py` supports:

- `--data_path`
  - default: `neutralizing-biased-phrase/src/bias_data/WNC/biased.full.test`
- `--few_shot_path`
  - default: `neutralizing-biased-phrase/src/bias_data/WNC/biased.full.train`
- `--tagger_ckpt`
  - default: `neutralizing-biased-phrase/src/train_tagging/biased_phrase_tagger.ckpt`
- `--output_dir`
  - default: `llm_prompting_runner/outputs`
- `--samples`
  - default: `200`
- `--seed`
  - default: `42`
- `--models` (one or more)
  - default: `meta-llama/llama-3.1-8b-instruct`
- `--strategies` (one or more)
  - default: `zero_shot few_shot with_bias_tags npov self_refine`
  - choices: `zero_shot`, `few_shot`, `with_bias_tags`, `npov`, `self_refine`
- `--few_shot_k`
  - default: `3`
- `--temperature`
  - default: `0.0`
- `--max_tokens`
  - default: `220`
- `--api_base`
  - default: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
  - for OpenRouter, pass: `--api_base https://openrouter.ai/api/v1`
- `--api_key`
  - default: empty
- `--api_key_file`
  - default: `llm_prompting_runner/.local.env`
- `--bert_vocab`
  - default: `neutralizing-biased-phrase/src/bias_data/bert.vocab`
- `--parallel_requests`
  - default: `1`
- `--request_timeout`
  - default: `60.0`
- `--allow_fallbacks / --no-allow_fallbacks`
  - default: `--allow_fallbacks`
- `--quality_retry_once / --no-quality_retry_once`
  - default: `--no-quality_retry_once`
- `--quality_retry_attempts`
  - default: `1`
- `--retries`
  - default: `3`

## Outputs

Written to `llm_prompting_runner/outputs/`:

- One JSONL per model/strategy run (sentence-level predictions + sample metrics)
- `summary.json`
- `summary.csv`

If you store runs separately for comparison, use:

- `llm_prompting_runner/outputs_qwen/`
- `llm_prompting_runner/outputs_llama/`
