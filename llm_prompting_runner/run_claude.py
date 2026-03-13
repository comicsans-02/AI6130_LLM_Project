"""
Run bias-neutralization prompting experiments using Claude models (Haiku, Sonnet, Opus)
via the `claude` CLI subprocess.

Adapts run_local_models.py — imports all shared utilities from it, only implements
the Claude-specific runner.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import threading
import time

# Claude CLI refuses to run inside a Claude Code session (CLAUDECODE env var).
# Strip it so subprocess calls work when launched from within Claude Code.
_SUBPROCESS_ENV = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from run_local_models import (
    REPO_ROOT,
    SCRIPT_DIR,
    PreparedExample,
    Example,
    LocalBiasTagger,
    WordPieceAdapter,
    load_wnc,
    load_wnc_rows,
    normalize_prediction,
    is_low_quality_rewrite,
    compute_metrics,
    bias_phrase_retention,
    edit_rate,
    wrap_bias_spans,
    simple_tokenize,
    safe_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "Task: Rewrite the biased input sentence to neutral point-of-view.\n"
    "Rules:\n"
    "1) Preserve factual meaning and named entities.\n"
    "2) Remove subjective, loaded, inflammatory, or opinionated wording.\n"
    "3) Keep edits minimal and targeted; do not add new claims.\n"
    "4) Keep dates, numbers, and named entities unchanged unless grammar requires it.\n"
    "5) Output exactly one rewritten sentence.\n"
    "6) Return only sentence text. No prefaces (e.g., 'Here is'), no explanations, no bullets, no quotes.\n"
    "7) Never output XML-like tags such as <bias> or </bias>."
)

MODEL_CONFIG: Dict[str, dict] = {
    "haiku": {
        "model_id": "haiku",
        "default_samples": 500,
        "default_strategies": ["zero_shot", "few_shot", "npov", "with_bias_tags"],
    },
    "sonnet": {
        "model_id": "sonnet",
        "default_samples": 500,
        "default_strategies": ["zero_shot", "few_shot", "npov", "with_bias_tags", "self_refine"],
    },
    "opus": {
        "model_id": "opus",
        "default_samples": 50,  # Small batches — increment --samples each week to extend
        "default_strategies": ["zero_shot", "npov"],
    },
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_claude_prompt(
    strategy: str,
    prepared_ex: PreparedExample,
    few_shots: List[Example],
) -> str:
    """
    Flatten the system+user messages into a single string for ``claude -p``.

    Claude CLI has no ``--system`` flag, so system content is embedded inline.
    """

    if strategy in ("zero_shot", "npov", "self_refine"):
        # self_refine uses the same initial prompt; refinement is a separate call.
        return REWRITE_SYSTEM + "\n\n" + prepared_ex.source

    if strategy == "few_shot":
        parts = [REWRITE_SYSTEM, ""]
        for ex in few_shots:
            parts.append(f"Original: {ex.source}\nNeutral: {ex.target}\n")
        parts.append(f"Now rewrite:\n{prepared_ex.source}")
        return "\n".join(parts)

    if strategy == "with_bias_tags":
        bias_instruction = (
            "Rewrite to neutral language while preserving factual meaning. "
            "Tokens inside <bias>...</bias> are likely biased and must be neutralized first. "
            "The tags are hints only; do not output any tags. "
            "Do not add new facts. Return exactly one sentence."
        )
        return (
            REWRITE_SYSTEM
            + "\n\n"
            + bias_instruction
            + "\n\n"
            + f"Sentence: {prepared_ex.tagged_source_for_prompt}"
        )

    # Fallback: treat unknown strategy same as zero_shot.
    return REWRITE_SYSTEM + "\n\n" + prepared_ex.source


# ---------------------------------------------------------------------------
# Claude CLI wrapper
# ---------------------------------------------------------------------------


def call_claude(prompt: str, model: str, retries: int = 3) -> str:
    """Call the ``claude`` CLI and return the stripped stdout."""
    delay = 5.0
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", model],
                capture_output=True,
                text=True,
                timeout=120,
                env=_SUBPROCESS_ENV,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return ""


# ---------------------------------------------------------------------------
# Self-refine helpers
# ---------------------------------------------------------------------------


def refine_once_claude(source: str, draft: str, model: str, retries: int = 3) -> str:
    """Two-step critique-then-improve via separate subprocess calls."""
    critique_prompt = (
        "Critique this rewrite for neutrality and meaning preservation. "
        "Keep critique concise in 1-2 sentences.\n\n"
        f"Source: {source}\nRewrite: {draft}"
    )
    critique = call_claude(critique_prompt, model, retries=retries)
    if not critique:
        return draft

    improve_prompt = (
        "Improve the rewrite using the critique. Return only the improved final sentence.\n\n"
        f"Source: {source}\nCurrent rewrite: {draft}\nCritique: {critique}"
    )
    improved = call_claude(improve_prompt, model, retries=retries)
    return improved if improved else draft


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def load_done_ids(jsonl_path: Path) -> set:
    """Return the set of IDs already present in an existing JSONL file."""
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    done.add(rec["id"])
            except json.JSONDecodeError:
                continue
    return done


def load_existing_summary(summary_path: Path) -> List[dict]:
    """Load previously saved summary rows so we can merge/update."""
    if not summary_path.exists():
        return []
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path).resolve()
    tagger_ckpt = Path(args.tagger_ckpt).resolve()

    # ------------------------------------------------------------------
    # 1. Load WNC data
    # ------------------------------------------------------------------
    # Load the full eval pool at the maximum sample count across all models.
    max_samples = max(
        args.samples if args.samples is not None else 0,
        max(cfg["default_samples"] for cfg in MODEL_CONFIG.values()),
    )
    all_eval_rows = load_wnc(data_path, max_samples, args.seed)

    few_rows = load_wnc_rows(Path(args.few_shot_path).resolve())
    eval_ids = {ex.idx for ex in all_eval_rows}
    few_pool = [ex for ex in few_rows if ex.idx not in eval_ids]
    if len(few_pool) < args.few_shot_k:
        raise RuntimeError(
            f"Few-shot pool too small after excluding eval overlap: "
            f"need {args.few_shot_k}, got {len(few_pool)}."
        )
    rng = random.Random(args.seed + 1)
    rng.shuffle(few_pool)
    few_shots = few_pool[: args.few_shot_k]

    # ------------------------------------------------------------------
    # 2. Prepare examples (run tagger upfront)
    # ------------------------------------------------------------------
    tagger = LocalBiasTagger(tagger_ckpt)
    sem_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    wp_tokenizer = WordPieceAdapter(Path(args.bert_vocab).resolve())

    all_prepared: List[PreparedExample] = []
    for ex in all_eval_rows:
        src_tokens = ex.source_wnc_tokens
        src_tags = tagger.predict_tags(src_tokens)
        src_bias_indices = tagger.bias_indices(src_tags)
        src_bias_count = len(src_bias_indices)
        src_bias_density = src_bias_count / max(1, len(src_tokens))
        tagged_source_for_prompt = wrap_bias_spans(src_tokens, src_tags)
        src_plain_tokens = simple_tokenize(ex.source.lower())
        ref_plain_tokens = simple_tokenize(ex.target.lower())
        ref_edit = edit_rate(src_plain_tokens, ref_plain_tokens)
        all_prepared.append(
            PreparedExample(
                idx=ex.idx,
                source=ex.source,
                target=ex.target,
                src_tokens=src_tokens,
                src_tags=src_tags,
                src_bias_indices=src_bias_indices,
                src_bias_count=src_bias_count,
                src_bias_density=src_bias_density,
                tagged_source_for_prompt=tagged_source_for_prompt,
                src_plain_tokens=src_plain_tokens,
                ref_plain_tokens=ref_plain_tokens,
                ref_edit_rate=ref_edit,
            )
        )

    if all(ex.src_bias_count == 0 for ex in all_prepared):
        raise RuntimeError(
            "Sanity check failed: all source_bias_count values are 0. "
            "Tagger/tokenization input is likely incompatible."
        )

    # ------------------------------------------------------------------
    # 3. Per-model / per-strategy loop
    # ------------------------------------------------------------------
    summary_json_path = out_dir / "summary.json"
    summary_csv_path = out_dir / "summary.csv"
    summary_rows = load_existing_summary(summary_json_path)

    for model_key in args.models:
        cfg = MODEL_CONFIG[model_key]
        model_id = cfg["model_id"]
        n_samples = args.samples if args.samples is not None else cfg["default_samples"]
        strategies = args.strategies if args.strategies is not None else cfg["default_strategies"]

        # Slice prepared rows to this model's sample count.
        prepared_eval_rows = all_prepared[:n_samples]

        for strategy in strategies:
            run_name = f"{safe_name(model_key)}__{strategy}"
            pred_path = out_dir / f"{run_name}.jsonl"
            dry_prefix = "[dry_run] " if args.dry_run else ""

            # --- Resume --------------------------------------------------
            if args.overwrite and pred_path.exists():
                pred_path.unlink()
                done_ids: set = set()
            else:
                done_ids = load_done_ids(pred_path)

            todo_rows = [ex for ex in prepared_eval_rows if ex.idx not in done_ids]
            if args.dry_run:
                todo_rows = todo_rows[:3]

            if done_ids:
                print(f"{dry_prefix}[skipped {len(done_ids)} already done] {run_name}")

            if not todo_rows:
                print(f"{dry_prefix}[done] {run_name} — nothing to do, all {len(done_ids)} already done")
                # Still need aggregate metrics for summary.
                # Read back all records from JSONL.
                pass  # handled below after the generation block

            # --- Generation ----------------------------------------------
            refs: List[str] = []
            preds: List[str] = []
            bias_retention_vals: List[float] = []
            bias_phrase_retention_vals: List[float] = []
            over_edit_vals: List[float] = []
            no_bias_reduction_count = 0
            write_lock = threading.Lock()

            file_mode = "a"  # append for resume
            with pred_path.open(file_mode, encoding="utf-8") as fout:

                def generate_one(ex: PreparedExample) -> Tuple[PreparedExample, str]:
                    prompt = build_claude_prompt(strategy, ex, few_shots)
                    pred_local = call_claude(prompt, model_id, retries=args.retries)
                    if not pred_local:
                        pred_local = ex.source

                    if strategy == "self_refine" and pred_local != ex.source:
                        pred_local = refine_once_claude(
                            ex.source, pred_local, model_id, retries=args.retries
                        )

                    pred_local = normalize_prediction(pred_local, ex.source)

                    # Quality retry once.
                    if is_low_quality_rewrite(pred_local, ex.source):
                        prompt2 = build_claude_prompt(strategy, ex, few_shots)
                        retry_text = call_claude(prompt2, model_id, retries=args.retries)
                        if retry_text:
                            retry_text = normalize_prediction(retry_text, ex.source)
                            if not is_low_quality_rewrite(retry_text, ex.source):
                                pred_local = retry_text

                    if is_low_quality_rewrite(pred_local, ex.source):
                        pred_local = ex.source

                    return ex, pred_local

                def process_result(ex: PreparedExample, pred: str) -> None:
                    nonlocal no_bias_reduction_count

                    pred_tokens = wp_tokenizer.tokenize(pred)
                    pred_tags = tagger.predict_tags(pred_tokens)
                    pred_bias = tagger.bias_indices(pred_tags)

                    src_bias_count = ex.src_bias_count
                    pred_bias_count = len(pred_bias)
                    src_bias_density = ex.src_bias_density
                    pred_bias_density = pred_bias_count / max(1, len(pred_tokens))

                    if src_bias_density > 0:
                        bias_retention = min(1.0, pred_bias_density / src_bias_density)
                        bias_retention_vals.append(bias_retention)
                    else:
                        bias_retention = 0.0

                    src_bias_phrase_count, retained_bias_phrase_count, bias_phrase_retention_value = (
                        bias_phrase_retention(ex.src_tokens, ex.src_tags, pred_tokens)
                    )
                    if src_bias_phrase_count > 0:
                        bias_phrase_retention_vals.append(bias_phrase_retention_value)

                    pred_plain_tokens = simple_tokenize(pred.lower())
                    pred_edit_rate = edit_rate(ex.src_plain_tokens, pred_plain_tokens)
                    over_edit_rate_val = max(0.0, pred_edit_rate - ex.ref_edit_rate)
                    over_edit_vals.append(over_edit_rate_val)

                    refs.append(ex.target)
                    preds.append(pred)

                    if src_bias_density > 0 and pred_bias_density >= src_bias_density:
                        no_bias_reduction_count += 1

                    record = {
                        "id": ex.idx,
                        "source": ex.source,
                        "reference": ex.target,
                        "prediction": pred,
                        "source_bias_count": src_bias_count,
                        "prediction_bias_count": pred_bias_count,
                        "bias_retention_sample": round(bias_retention, 4),
                        "source_bias_phrase_count": src_bias_phrase_count,
                        "retained_bias_phrase_count": retained_bias_phrase_count,
                        "bias_phrase_retention_sample": round(bias_phrase_retention_value, 4),
                        "over_edit_rate_sample": round(over_edit_rate_val, 4),
                    }
                    with write_lock:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()

                if todo_rows:
                    if args.parallel_workers > 1:
                        with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
                            for ex, pred in tqdm(
                                pool.map(generate_one, todo_rows),
                                total=len(todo_rows),
                                desc=f"{dry_prefix}{run_name}",
                                leave=True,
                            ):
                                process_result(ex, pred)
                    else:
                        for ex in tqdm(
                            todo_rows,
                            desc=f"{dry_prefix}{run_name}",
                            leave=True,
                        ):
                            ex, pred = generate_one(ex)
                            process_result(ex, pred)

            # --- Aggregate metrics from ALL records (including resumed) ---
            # Re-read the complete JSONL to compute aggregate metrics.
            all_refs: List[str] = []
            all_preds: List[str] = []
            all_bias_retention_vals: List[float] = []
            all_bias_phrase_retention_vals: List[float] = []
            all_over_edit_vals: List[float] = []
            all_no_bias_reduction_count = 0

            if pred_path.exists():
                with pred_path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        all_refs.append(rec["reference"])
                        all_preds.append(rec["prediction"])
                        if rec.get("source_bias_count", 0) > 0:
                            all_bias_retention_vals.append(rec["bias_retention_sample"])
                        if rec.get("source_bias_phrase_count", 0) > 0:
                            all_bias_phrase_retention_vals.append(rec["bias_phrase_retention_sample"])
                        all_over_edit_vals.append(rec.get("over_edit_rate_sample", 0.0))
                        if rec.get("source_bias_count", 0) > 0:
                            # bias_retention_sample >= 1.0 means pred density >= src density
                            if rec["bias_retention_sample"] >= 1.0:
                                all_no_bias_reduction_count += 1

            if all_refs and all_preds:
                metrics = compute_metrics(all_refs, all_preds, sem_model)
            else:
                metrics = {
                    "BLEU": 0.0,
                    "Token-Level Accuracy": 0.0,
                    "SemanticSimilarity": 0.0,
                    "BERTScoreF1": 0.0,
                    "METEOR": 0.0,
                }

            metrics["BiasRetentionRate"] = round(
                float(np.mean(all_bias_retention_vals)) if all_bias_retention_vals else 0.0, 4
            )
            metrics["BiasPhraseRetentionRate"] = round(
                float(np.mean(all_bias_phrase_retention_vals)) if all_bias_phrase_retention_vals else 0.0, 4
            )
            metrics["OverEditRate"] = round(
                float(np.mean(all_over_edit_vals)) if all_over_edit_vals else 0.0, 4
            )
            metrics["NoBiasReductionCount"] = all_no_bias_reduction_count
            metrics["n"] = len(all_preds)
            metrics["model"] = model_key
            metrics["strategy"] = strategy

            # Merge into summary: update existing row for same model+strategy, or append.
            updated = False
            for i, row in enumerate(summary_rows):
                if row.get("model") == model_key and row.get("strategy") == strategy:
                    summary_rows[i] = metrics
                    updated = True
                    break
            if not updated:
                summary_rows.append(metrics)

            print(f"{dry_prefix}[done] {run_name} -> {pred_path}")

    # ------------------------------------------------------------------
    # 4. Write summary files
    # ------------------------------------------------------------------
    if summary_rows:
        summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        all_keys = list(summary_rows[0].keys())
        with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary: {summary_json_path}")
        print(f"Saved summary: {summary_csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bias-neutralization prompting experiments via Claude CLI."
    )
    parser.add_argument(
        "--data_path",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "bias_data" / "WNC" / "biased.full.test"),
    )
    parser.add_argument(
        "--few_shot_path",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "bias_data" / "WNC" / "biased.full.train"),
    )
    parser.add_argument(
        "--tagger_ckpt",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "train_tagging" / "biased_phrase_tagger.ckpt"),
    )
    parser.add_argument(
        "--bert_vocab",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "bias_data" / "bert.vocab"),
    )
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "outputs_claude"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["haiku", "sonnet"],
        choices=list(MODEL_CONFIG.keys()),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=["zero_shot", "few_shot", "with_bias_tags", "npov", "self_refine"],
        help="Override per-model default strategies. If omitted, uses MODEL_CONFIG defaults.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override per-model default sample count. If omitted, uses MODEL_CONFIG defaults.",
    )
    parser.add_argument("--few_shot_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel_workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 3 examples per strategy for quick testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing JSONL and start fresh.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
