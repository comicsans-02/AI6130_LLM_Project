import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from run_local_models import (
    BERT_VOCAB_PATH,
    REPO_ROOT,
    LocalBiasTagger,
    WordPieceAdapter,
    bias_phrase_retention,
    compute_metrics,
    edit_rate,
    load_wnc_rows,
    simple_tokenize,
)


def parse_model_strategy(file_path: Path) -> Tuple[str, str]:
    stem = file_path.stem
    if "__" not in stem:
        return stem, "unknown"
    model, strategy = stem.rsplit("__", 1)
    return model, strategy


def load_source_token_index(data_path: Path) -> Dict[str, List[str]]:
    rows = load_wnc_rows(data_path)
    return {r.idx: r.source_wnc_tokens for r in rows}


def rescore_file_inplace(
    file_path: Path,
    tagger: LocalBiasTagger,
    wp_tokenizer: WordPieceAdapter,
    source_token_index: Dict[str, List[str]],
    sem_model: SentenceTransformer,
) -> Dict[str, float]:
    refs: List[str] = []
    preds: List[str] = []
    bias_retention_vals: List[float] = []
    bias_phrase_retention_vals: List[float] = []
    over_edit_vals: List[float] = []
    no_bias_reduction_count = 0
    row_count = 0

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with file_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            row_count += 1

            ex_id = str(row.get("id", ""))
            source = str(row.get("source", ""))
            reference = str(row.get("reference", ""))
            prediction = str(row.get("prediction", ""))

            src_tokens = source_token_index.get(ex_id)
            if not src_tokens:
                # Fallback if ID is missing from dataset map.
                src_tokens = wp_tokenizer.tokenize(source)

            pred_tokens = wp_tokenizer.tokenize(prediction)
            src_tags = tagger.predict_tags(src_tokens)
            pred_tags = tagger.predict_tags(pred_tokens)

            src_bias = tagger.bias_indices(src_tags)
            pred_bias = tagger.bias_indices(pred_tags)

            src_bias_count = len(src_bias)
            pred_bias_count = len(pred_bias)
            src_bias_density = src_bias_count / max(1, len(src_tokens))
            pred_bias_density = pred_bias_count / max(1, len(pred_tokens))

            if src_bias_density > 0:
                bias_retention = min(1.0, pred_bias_density / src_bias_density)
                bias_retention_vals.append(bias_retention)
            else:
                bias_retention = 0.0

            src_bias_phrase_count, retained_bias_phrase_count, phrase_retention = bias_phrase_retention(
                src_tokens, src_tags, pred_tokens
            )
            if src_bias_phrase_count > 0:
                bias_phrase_retention_vals.append(phrase_retention)

            src_plain = simple_tokenize(source.lower())
            ref_plain = simple_tokenize(reference.lower())
            pred_plain = simple_tokenize(prediction.lower())
            ref_edit_rate = edit_rate(src_plain, ref_plain)
            pred_edit_rate = edit_rate(src_plain, pred_plain)
            over_edit_rate = max(0.0, pred_edit_rate - ref_edit_rate)
            over_edit_vals.append(over_edit_rate)

            if src_bias_density > 0 and pred_bias_density >= src_bias_density:
                no_bias_reduction_count += 1

            row["source_bias_count"] = src_bias_count
            row["prediction_bias_count"] = pred_bias_count
            row["bias_retention_sample"] = round(bias_retention, 4)
            row["source_bias_phrase_count"] = src_bias_phrase_count
            row["retained_bias_phrase_count"] = retained_bias_phrase_count
            row["bias_phrase_retention_sample"] = round(phrase_retention, 4)
            row["over_edit_rate_sample"] = round(over_edit_rate, 4)

            refs.append(reference)
            preds.append(prediction)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp_path.replace(file_path)

    metrics = compute_metrics(refs, preds, sem_model)
    model_name, strategy = parse_model_strategy(file_path)
    metrics["BiasRetentionRate"] = round(sum(bias_retention_vals) / max(1, len(bias_retention_vals)), 4)
    metrics["BiasPhraseRetentionRate"] = round(
        sum(bias_phrase_retention_vals) / max(1, len(bias_phrase_retention_vals)), 4
    )
    metrics["OverEditRate"] = round(sum(over_edit_vals) / max(1, len(over_edit_vals)), 4)
    metrics["NoBiasReductionCount"] = no_bias_reduction_count
    metrics["n"] = row_count
    metrics["model"] = model_name
    metrics["strategy"] = strategy
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore existing JSONL outputs in place without calling any generation API."
    )
    parser.add_argument("--input_dir", default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--pattern", default="*__*.jsonl")
    parser.add_argument(
        "--data_path",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "bias_data" / "WNC" / "biased.full.test"),
    )
    parser.add_argument(
        "--tagger_ckpt",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "train_tagging" / "biased_phrase_tagger.ckpt"),
    )
    parser.add_argument("--bert_vocab", default=str(BERT_VOCAB_PATH))
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    files = sorted([p for p in input_dir.glob(args.pattern) if p.is_file()])
    if not files:
        raise RuntimeError(f"No files matched pattern '{args.pattern}' in {input_dir}")

    print(f"Rescoring {len(files)} files in place...")
    tagger = LocalBiasTagger(Path(args.tagger_ckpt).resolve())
    wp_tokenizer = WordPieceAdapter(Path(args.bert_vocab).resolve())
    source_token_index = load_source_token_index(Path(args.data_path).resolve())
    sem_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    summary_rows = []
    for file_path in tqdm(files, desc="files", leave=True):
        metrics = rescore_file_inplace(
            file_path=file_path,
            tagger=tagger,
            wp_tokenizer=wp_tokenizer,
            source_token_index=source_token_index,
            sem_model=sem_model,
        )
        summary_rows.append(metrics)
        print(f"[rescored] {file_path.name}")

    summary_json = input_dir / "summary.json"
    summary_csv = input_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nUpdated in place.")
    print(f"Summary: {summary_json}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
