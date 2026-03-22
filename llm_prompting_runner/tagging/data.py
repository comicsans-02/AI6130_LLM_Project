import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter
import os
import json
import linecache
from multiprocessing import Pool

class TaggingDataset(Dataset):
    def __init__(self, sequences, labels, tok2id, label2id, max_len=128):
        self.sequences = sequences
        self.labels = labels
        self.tok2id = tok2id
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx][:self.max_len]
        labels = self.labels[idx][:self.max_len]

        input_ids = [self.tok2id.get(tok, self.tok2id['<unk>']) for tok in tokens]
        label_ids = [self.label2id.get(label, self.label2id['O']) for label in labels]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'length': len(tokens)
        }

def tag_with_phrases(token_list, bias_phrase_tuples):
    tags = ['O'] * len(token_list)
    for phrase in bias_phrase_tuples:
        n = len(phrase)
        for i in range(len(token_list) - n + 1):
            if token_list[i:i+n] == list(phrase):
                tags[i] = 'B'
                for j in range(1, n):
                    tags[i+j] = 'I'
    return tags

def extract_biased_phrases_from_wnc(filepath, min_freq=1, max_phrase_len=6, cache_file="train_tagging/bias_phrases_cache.json"):
    # Check for cached phrases
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Check if file is not empty
                    cached_phrases = json.loads(content)
                    # Convert cached phrases to Counter
                    return Counter({tuple(phrase): count for phrase, count in cached_phrases})
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load cache file {cache_file}: {e}. Rebuilding cache.")

    phrase_counter = Counter()
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            orig = parts[1].strip().split()
            edited = parts[2].strip().split()

            if abs(len(orig) - len(edited)) > 10:
                continue

            i = 0
            while i < min(len(orig), len(edited)) and orig[i] == edited[i]:
                i += 1
            j = 0
            while i + j < len(orig) and i + j < len(edited) and orig[-1 - j] == edited[-1 - j]:
                j += 1

            if i <= len(orig) - j - 1:
                biased_span = orig[i:len(orig) - j]
                if 0 < len(biased_span) <= max_phrase_len:
                    if any(c.isalnum() for c in ' '.join(biased_span)):
                        phrase_counter[tuple(biased_span)] += 1

    # Cache the results
    filtered_phrases = [(list(k), v) for k, v in phrase_counter.items() if v >= min_freq]
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_phrases, f)
    except Exception as e:
        print(f"Warning: Failed to write cache file {cache_file}: {e}")

    return Counter({tuple(k): v for k, v in filtered_phrases})

def process_line(line_data):
    line, bias_phrases = line_data
    if '\t' not in line:
        return None
    parts = line.strip().split('\t')
    if len(parts) < 3:
        return None
    tokens = parts[1].strip().split()
    tag_seq = tag_with_phrases(tokens, bias_phrases)
    return tokens, tag_seq

def load_tagging_data_from_wiki_format(filepath, bias_phrases=None, num_workers=16):
    if bias_phrases is None:
        bias_phrases = extract_biased_phrases_from_wnc(filepath)
    
    sequences, labels = [], []
    
    # Read all lines first
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()
    
    # Prepare data for multiprocessing
    line_data = [(line, bias_phrases) for line in lines]
    
    # Process lines in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_line, line_data)
    
    # Filter out None results and collect sequences and labels
    for result in results:
        if result is not None:
            tokens, tag_seq = result
            sequences.append(tokens)
            labels.append(tag_seq)
    
    return sequences, labels, bias_phrases