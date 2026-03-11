import argparse
import csv
import json
import os
import random
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sacrebleu import corpus_bleu
from sentence_transformers import SentenceTransformer, util
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `openai`. Install it with: uv pip install openai"
    ) from exc

try:
    from bert_score import score as bertscore_score

    HAS_BERTSCORE = True
except Exception:
    HAS_BERTSCORE = False

try:
    from nltk.translate.meteor_score import meteor_score

    HAS_METEOR = True
except Exception:
    HAS_METEOR = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "neutralizing-biased-phrase" / "src"
BERT_VOCAB_PATH = SRC_DIR / "bias_data" / "bert.vocab"

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tagging.model import BiasedPhraseTagger  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    module=r"bert_score\.score",
)
warnings.filterwarnings(
    "ignore",
    message=r"Some weights of .* were not initialized from the model checkpoint.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Some weights of .* were not used when initializing.*",
)
try:
    from transformers.utils import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    pass


def safe_name(value: str) -> str:
    # Make provider/model slugs filesystem-safe for output filenames.
    return re.sub(r'[\\/:*?"<>|]+', "_", value)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def detok(tokens: List[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '" ', text)
    return text.strip()


def detok_wordpiece(tokens: List[str]) -> str:
    if not tokens:
        return ""
    out = []
    for tok in tokens:
        if tok.startswith("##") and out:
            out[-1] = out[-1] + tok[2:]
        else:
            out.append(tok)
    return detok(out)


def wrap_bias_spans(tokens: List[str], tags: List[str]) -> str:
    out = []
    in_bias = False
    for tok, tag in zip(tokens, tags):
        if tag == "B":
            if in_bias:
                out.append("</bias>")
            out.append("<bias>")
            out.append(tok)
            in_bias = True
        elif tag == "I":
            out.append(tok)
        else:
            if in_bias:
                out.append("</bias>")
                in_bias = False
            out.append(tok)
    if in_bias:
        out.append("</bias>")
    return detok_wordpiece(out)


def extract_bias_spans(tokens: List[str], tags: List[str]) -> List[List[str]]:
    spans: List[List[str]] = []
    current: List[str] = []
    for tok, tag in zip(tokens, tags):
        if tag == "B":
            if current:
                spans.append(current)
            current = [tok]
        elif tag == "I":
            if current:
                current.append(tok)
            else:
                current = [tok]
        else:
            if current:
                spans.append(current)
                current = []
    if current:
        spans.append(current)
    return spans


def has_subsequence(haystack: List[str], needle: List[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    if n > len(haystack):
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def bias_phrase_retention(
    src_tokens: List[str], src_tags: List[str], pred_tokens: List[str]
) -> Tuple[int, int, float]:
    spans = extract_bias_spans(src_tokens, src_tags)
    total = len(spans)
    if total == 0:
        return 0, 0, 0.0
    retained = sum(1 for span in spans if has_subsequence(pred_tokens, span))
    return total, retained, retained / total


@dataclass
class Example:
    idx: str
    source: str
    target: str
    source_wnc_tokens: List[str]


@dataclass
class PreparedExample:
    idx: str
    source: str
    target: str
    src_tokens: List[str]
    src_tags: List[str]
    src_bias_indices: set
    src_bias_count: int
    src_bias_density: float
    tagged_source_for_prompt: str
    src_plain_tokens: List[str]
    ref_plain_tokens: List[str]
    ref_edit_rate: float


class WordPieceAdapter:
    def __init__(self, vocab_path: Path):
        # Use packaged tokenizer implementation for compatibility and stability.
        self.tokenizer = BertWordPieceTokenizer(
            vocab=str(vocab_path),
            lowercase=True,
        )

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text, add_special_tokens=False).tokens


class LocalBiasTagger:
    def __init__(self, ckpt_path: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device)
        self.tok2id = ckpt["tok2id"]
        self.id2label = {v: k for k, v in ckpt["label2id"].items()}
        self.device = device
        self.model = BiasedPhraseTagger(
            vocab_size=len(self.tok2id),
            embedding_dim=128,
            hidden_dim=256,
            num_labels=len(self.id2label),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict_tags(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        ids = [self.tok2id.get(t, self.tok2id["<unk>"]) for t in tokens]
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids, lengths)
            pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return [self.id2label[p] for p in pred[: len(tokens)]]

    def bias_indices(self, tags: List[str]) -> set:
        return {i for i, t in enumerate(tags) if t in {"B", "I"}}


def load_wnc_rows(path: Path) -> List[Example]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            idx = parts[0]
            source_wnc_tokens = parts[1].strip().split() if len(parts) > 1 else []
            source = parts[3] if len(parts) > 4 else parts[1]
            target = parts[4] if len(parts) > 4 else parts[2]
            source = source.strip()
            target = target.strip()
            if source and target and source_wnc_tokens:
                rows.append(Example(idx=idx, source=source, target=target, source_wnc_tokens=source_wnc_tokens))
    return rows


def load_wnc(path: Path, max_examples: int, seed: int) -> List[Example]:
    rows = load_wnc_rows(path)

    rng = random.Random(seed)
    rng.shuffle(rows)

    eval_count = min(max_examples, len(rows))
    return rows[:eval_count]


def build_messages(
    strategy: str,
    prepared_ex: PreparedExample,
    few_shots: List[Example],
) -> List[Dict[str, str]]:
    rewrite_system = (
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

    if strategy == "npov":
        return [{"role": "system", "content": rewrite_system}, {"role": "user", "content": prepared_ex.source}]

    if strategy == "with_bias_tags":
        user = (
            "Rewrite to neutral language while preserving factual meaning. "
            "Tokens inside <bias>...</bias> are likely biased and must be neutralized first. "
            "The tags are hints only; do not output any tags. "
            "Do not add new facts. Return exactly one sentence.\n\n"
            f"Sentence: {prepared_ex.tagged_source_for_prompt}"
        )
        return [{"role": "system", "content": rewrite_system}, {"role": "user", "content": user}]

    if strategy == "few_shot":
        messages = [
            {
                "role": "system",
                "content": rewrite_system,
            }
        ]
        for ex in few_shots:
            messages.append({"role": "user", "content": ex.source})
            messages.append({"role": "assistant", "content": ex.target})
        messages.append({"role": "user", "content": prepared_ex.source})
        return messages

    return [
        {
            "role": "system",
            "content": rewrite_system,
        },
        {"role": "user", "content": prepared_ex.source},
    ]


def call_openai_chat(
    client: "OpenAI",
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    allow_fallbacks: bool,
) -> str:
    extra_body = {"provider": {"allow_fallbacks": allow_fallbacks}}

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content
    return (content or "").strip()


def call_openai_with_retry(
    client: "OpenAI",
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    allow_fallbacks: bool,
    retries: int = 3,
) -> str:
    delay = 1.0
    last_exc = None
    for _ in range(retries):
        try:
            return call_openai_chat(
                client,
                model,
                messages,
                temperature,
                max_tokens,
                request_timeout=request_timeout,
                allow_fallbacks=allow_fallbacks,
            )
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"OpenAI-compatible API call failed after {retries} attempts: {last_exc}")


def refine_once(
    client: "OpenAI",
    model: str,
    source: str,
    draft: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    allow_fallbacks: bool,
    retries: int,
) -> str:
    critique_prompt = (
        "Critique this rewrite for neutrality and meaning preservation. "
        "Keep critique concise in 1-2 sentences.\n\n"
        f"Source: {source}\nRewrite: {draft}"
    )
    critique = call_openai_with_retry(
        client,
        model,
        [{"role": "user", "content": critique_prompt}],
        temperature,
        max_tokens,
        request_timeout=request_timeout,
        allow_fallbacks=allow_fallbacks,
        retries=retries,
    )
    improve_prompt = (
        "Improve the rewrite using the critique. Return only the improved final sentence.\n\n"
        f"Source: {source}\nCurrent rewrite: {draft}\nCritique: {critique}"
    )
    return call_openai_with_retry(
        client,
        model,
        [{"role": "user", "content": improve_prompt}],
        temperature,
        max_tokens,
        request_timeout=request_timeout,
        allow_fallbacks=allow_fallbacks,
        retries=retries,
    )


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    if args.api_key_file:
        p = Path(args.api_key_file).resolve()
        if p.exists():
            raw = p.read_text(encoding="utf-8").strip()
            if "=" in raw and "\n" not in raw:
                _, val = raw.split("=", 1)
                return val.strip().strip('"').strip("'")
            if raw:
                return raw
    raise RuntimeError(
        "No API key found. Use --api_key, set DASHSCOPE_API_KEY/OPENAI_API_KEY, "
        "or place key in --api_key_file."
    )


def normalize_prediction(text: str, source_fallback: str) -> str:
    s = (text or "").strip()
    if not s:
        return source_fallback

    # Remove markdown fences/labels if the model emits formatted output.
    s = s.replace("```", " ").replace("Rewritten:", " ").replace("Rewrite:", " ").strip()
    s = re.sub(
        r"^\s*(here is|the sentence|certainly|i apologize|it seems like|this sentence)\b[^:]*:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = " ".join(s.split())

    s = s.strip(" \"'`")
    return s if s else source_fallback


def is_garbled_output(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    # Frequent replacement characters/question-mark runs are usually corrupted output.
    if "�" in s:
        return True
    if re.search(r"\?{6,}", s):
        return True
    return False


def is_low_quality_rewrite(prediction: str, source: str) -> bool:
    s = (prediction or "").strip()
    if is_garbled_output(s):
        return True
    if re.match(
        r"^\s*(it seems like|here is|the sentence|certainly|i cannot|i apologize|without additional context|if you're|the phrase you provided)\b",
        s,
        flags=re.IGNORECASE,
    ):
        return True
    if "<bias>" in s or "</bias>" in s:
        return True
    src_len = len(source.strip())
    pred_len = len(s)
    if src_len >= 50 and pred_len < int(src_len * 0.55):
        return True
    if src_len >= 40 and pred_len > int(src_len * 2.2):
        return True
    return False


def edit_rate(a_tokens: List[str], b_tokens: List[str]) -> float:
    sm = SequenceMatcher(a=a_tokens, b=b_tokens)
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            edits += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            edits += i2 - i1
        elif tag == "insert":
            edits += j2 - j1
    return edits / max(1, len(a_tokens))


def compute_metrics(
    references: List[str],
    predictions: List[str],
    semantic_model: SentenceTransformer,
) -> Dict[str, float]:
    bleu = corpus_bleu(predictions, [references]).score / 100.0
    token_acc = (
        sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        / max(1, len(predictions))
    )

    ref_emb = semantic_model.encode(references, convert_to_tensor=True, show_progress_bar=False)
    pred_emb = semantic_model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    sem = util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()

    result = {
        "BLEU": round(bleu, 4),
        "Token-Level Accuracy": round(float(token_acc), 4),
        "SemanticSimilarity": round(float(sem), 4),
    }

    if HAS_BERTSCORE:
        _, _, f1 = bertscore_score(
            predictions,
            references,
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
        result["BERTScoreF1"] = round(float(f1.mean().item()), 4)
    else:
        result["BERTScoreF1"] = float("nan")

    if HAS_METEOR:
        meteor_vals = [
            meteor_score([simple_tokenize(ref)], simple_tokenize(pred))
            for ref, pred in zip(references, predictions)
        ]
        result["METEOR"] = round(float(np.mean(meteor_vals)), 4)
    else:
        result["METEOR"] = float("nan")

    return result


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path).resolve()
    tagger_ckpt = Path(args.tagger_ckpt).resolve()

    eval_rows = load_wnc(data_path, args.samples, args.seed)
    few_rows = load_wnc_rows(Path(args.few_shot_path).resolve())
    eval_ids = {ex.idx for ex in eval_rows}
    few_pool = [ex for ex in few_rows if ex.idx not in eval_ids]
    if len(few_pool) < args.few_shot_k:
        raise RuntimeError(
            f"Few-shot pool too small after excluding eval overlap: "
            f"need {args.few_shot_k}, got {len(few_pool)}."
        )
    rng = random.Random(args.seed + 1)
    rng.shuffle(few_pool)
    few_shots = few_pool[: args.few_shot_k]

    tagger = LocalBiasTagger(tagger_ckpt)
    sem_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    wp_tokenizer = WordPieceAdapter(Path(args.bert_vocab).resolve())
    api_key = load_api_key(args)

    prepared_eval_rows: List[PreparedExample] = []
    for ex in eval_rows:
        # Use original WNC tokenization for source-side tagging compatibility.
        src_tokens = ex.source_wnc_tokens
        src_tags = tagger.predict_tags(src_tokens)
        src_bias_indices = tagger.bias_indices(src_tags)
        src_bias_count = len(src_bias_indices)
        src_bias_density = src_bias_count / max(1, len(src_tokens))
        tagged_source_for_prompt = wrap_bias_spans(src_tokens, src_tags)
        src_plain_tokens = simple_tokenize(ex.source.lower())
        ref_plain_tokens = simple_tokenize(ex.target.lower())
        ref_edit_rate = edit_rate(src_plain_tokens, ref_plain_tokens)
        prepared_eval_rows.append(
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
                ref_edit_rate=ref_edit_rate,
            )
        )

    if all(ex.src_bias_count == 0 for ex in prepared_eval_rows):
        raise RuntimeError(
            "Sanity check failed: all source_bias_count values are 0. "
            "Tagger/tokenization input is likely incompatible."
        )

    summary_rows = []
    for model_name in args.models:
        for strategy in args.strategies:
            run_name = f"{safe_name(model_name)}__{strategy}"
            pred_path = out_dir / f"{run_name}.jsonl"

            refs = []
            preds = []
            bias_retention_vals = []
            bias_phrase_retention_vals = []
            over_edit_vals = []
            no_bias_reduction_count = 0
            thread_state = threading.local()

            with pred_path.open("w", encoding="utf-8") as fout:
                def generate_one(ex: PreparedExample) -> Tuple[PreparedExample, str]:
                    if not hasattr(thread_state, "client"):
                        thread_state.client = OpenAI(api_key=api_key, base_url=args.api_base)
                    local_client = thread_state.client
                    quality_retries = max(0, args.quality_retry_attempts)
                    if args.quality_retry_once:
                        quality_retries = max(quality_retries, 1)
                    pred_local = ex.source
                    for retry_idx in range(quality_retries + 1):
                        msgs = build_messages(strategy, ex, few_shots)
                        if retry_idx > 0:
                            msgs = msgs + [
                                {
                                    "role": "user",
                                    "content": (
                                        "Output format correction: return exactly one rewritten sentence text only. "
                                        "No explanation, no assistant commentary, no list."
                                    ),
                                }
                            ]
                        pred_local = call_openai_with_retry(
                            client=local_client,
                            model=model_name,
                            messages=msgs,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            request_timeout=args.request_timeout,
                            allow_fallbacks=args.allow_fallbacks,
                            retries=args.retries,
                        )
                        if strategy == "self_refine":
                            pred_local = refine_once(
                                client=local_client,
                                model=model_name,
                                source=ex.source,
                                draft=pred_local,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                                allow_fallbacks=args.allow_fallbacks,
                                retries=args.retries,
                            )
                        pred_local = normalize_prediction(pred_local, ex.source)
                        if not is_low_quality_rewrite(pred_local, ex.source):
                            break
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

                    src_bias_phrase_count, retained_bias_phrase_count, bias_phrase_retention_value = bias_phrase_retention(
                        ex.src_tokens, ex.src_tags, pred_tokens
                    )
                    if src_bias_phrase_count > 0:
                        bias_phrase_retention_vals.append(bias_phrase_retention_value)

                    pred_plain_tokens = simple_tokenize(pred.lower())
                    pred_edit_rate = edit_rate(ex.src_plain_tokens, pred_plain_tokens)
                    over_edit_rate = max(0.0, pred_edit_rate - ex.ref_edit_rate)
                    over_edit_vals.append(over_edit_rate)

                    refs.append(ex.target)
                    preds.append(pred)

                    if src_bias_density > 0 and pred_bias_density >= src_bias_density:
                        no_bias_reduction_count += 1

                    fout.write(
                        json.dumps(
                            {
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
                                "over_edit_rate_sample": round(over_edit_rate, 4),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if args.parallel_requests > 1:
                    with ThreadPoolExecutor(max_workers=args.parallel_requests) as pool:
                        for ex, pred in tqdm(
                            pool.map(generate_one, prepared_eval_rows),
                            total=len(prepared_eval_rows),
                            desc=run_name,
                            leave=True,
                        ):
                            process_result(ex, pred)
                else:
                    for ex, pred in tqdm(
                        map(generate_one, prepared_eval_rows),
                        total=len(prepared_eval_rows),
                        desc=run_name,
                        leave=True,
                    ):
                        process_result(ex, pred)

            metrics = compute_metrics(refs, preds, sem_model)
            metrics["BiasRetentionRate"] = round(
                float(np.mean(bias_retention_vals)) if bias_retention_vals else 0.0, 4
            )
            metrics["BiasPhraseRetentionRate"] = round(
                float(np.mean(bias_phrase_retention_vals)) if bias_phrase_retention_vals else 0.0, 4
            )
            metrics["OverEditRate"] = round(float(np.mean(over_edit_vals)) if over_edit_vals else 0.0, 4)
            metrics["n"] = len(eval_rows)
            metrics["model"] = model_name
            metrics["strategy"] = strategy
            metrics["NoBiasReductionCount"] = no_bias_reduction_count
            summary_rows.append(metrics)

            print(f"[done] {run_name} -> {pred_path}")

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {summary_json}")
    print(f"Saved summary: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompting experiments via OpenAI SDK (Alibaba Cloud OpenAI-compatible endpoint)."
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
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/llama-3.1-8b-instruct"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "with_bias_tags", "npov", "self_refine"],
        choices=["zero_shot", "few_shot", "with_bias_tags", "npov", "self_refine"],
    )
    parser.add_argument("--few_shot_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=220)
    parser.add_argument("--api_base", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_key_file", default=str(SCRIPT_DIR / ".local.env"))
    parser.add_argument("--bert_vocab", default=str(BERT_VOCAB_PATH))
    parser.add_argument("--parallel_requests", type=int, default=1)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument(
        "--allow_fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quality_retry_once",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--quality_retry_attempts", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
