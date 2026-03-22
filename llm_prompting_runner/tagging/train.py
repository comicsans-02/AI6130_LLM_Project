import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tagging.data import load_tagging_data_from_wiki_format, extract_biased_phrases_from_wnc, TaggingDataset
from tagging.model import BiasedPhraseTagger
from tagging.utils import evaluate

from collections import Counter

def build_vocab(sequences, min_freq=1):
    counter = Counter()
    for seq in sequences:
        counter.update(seq)
    tok2id = {'<pad>': 0, '<unk>': 1}
    for tok, freq in counter.items():
        if freq >= min_freq:
            tok2id[tok] = len(tok2id)
    return tok2id

def build_token_vocab(sequences, min_freq=1):
    from collections import Counter
    counter = Counter()
    for seq in sequences:
        counter.update(seq)
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def collate_batch(batch):
    batch.sort(key=lambda x: x['length'], reverse=True)
    lengths = torch.tensor([b['length'] for b in batch])
    max_len = lengths[0]

    input_ids = [b['input_ids'][:max_len] for b in batch]
    labels = [b['labels'][:max_len] for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return input_ids, labels, lengths

def train(train_file, test_file, batch_size, epochs, model_out):
    bias_phrases = extract_biased_phrases_from_wnc(train_file)
    bias_phrases = Counter({
        tuple(eval(k)) if isinstance(k, str) and k.startswith("(") else k: v
        for k, v in bias_phrases.items()
    })

    print(f"Start loading & tagging training dataset...")
    train_sequences, train_labels, _ = load_tagging_data_from_wiki_format(train_file, bias_phrases)
    print(f"Start loading & tagging test dataset...")
    test_sequences, test_labels, _ = load_tagging_data_from_wiki_format(test_file, bias_phrases)

    label_counts = Counter([l for label_seq in train_labels for l in label_seq])
    print("Train label distribution:", label_counts)
    if label_counts.get('B', 0) == 0 and label_counts.get('I', 0) == 0:
        print("Warning: No biased phrases tagged. Check data and tagging logic.")
        
    print(f"# Biased phrases extracted: {len(bias_phrases)}")
    print("Top 10 extracted multi-word biased phrases:")
    multi_word_phrases = [(p, c) for p, c in bias_phrases.items() if len(p) > 1]
    for phrase, count in sorted(multi_word_phrases, key=lambda x: -x[1])[:10]:
        print(f" - {' '.join(phrase)}: {count}")

    print("Building vocab...")
    tok2id = build_token_vocab(train_sequences + test_sequences)
    label2id = {'O': 0, 'B': 1, 'I': 2}

    print("Preparing datasets...")
    train_dataset = TaggingDataset(train_sequences, train_labels, tok2id, label2id)
    test_dataset = TaggingDataset(test_sequences, test_labels, tok2id, label2id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiasedPhraseTagger(vocab_size=len(tok2id), embedding_dim=128, hidden_dim=256, num_labels=len(label2id))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    total_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = torch.tensor(batch['length'], dtype=torch.long).cpu()  # lengths must be on CPU

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        val_loss, acc, precision, recall, f1 = evaluate(model, test_loader, label2id, device)
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Acc: {acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        if f1 > 0.9600:
            print(f"[Early Stop] F1-score {f1:.4f} exceeded threshold 0.9600 at epoch {epoch}.")
            break

    total_time = time.time() - total_start
    print(f"Training completed. Total Time: {total_time:.2f}s")

    print(f"Saving model to {model_out}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tok2id': tok2id,
        'label2id': label2id
    }, model_out)

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids = []
    labels = []
    lengths = []
    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(item['input_ids'].tolist() + [0] * pad_len)
        labels.append(item['labels'].tolist() + [-1] * pad_len)
        lengths.append(item['length'])
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'length': lengths
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_out', required=True)
    args = parser.parse_args()

    train(
        train_file=args.train,
        test_file=args.test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_out=args.model_out
    )
