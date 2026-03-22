import torch
from sklearn.metrics import precision_recall_fscore_support

def evaluate(model, dataloader, label2id, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = torch.tensor(batch['length'], dtype=torch.long).cpu()

            logits = model(input_ids, lengths)  #(batch, seq_len, num_labels)
            logits = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            loss = criterion(logits, labels_flat)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_flat.cpu().tolist())

            correct += (preds == labels_flat).sum().item()
            total += (labels_flat != -1).sum().item()

    id2label = {v: k for k, v in label2id.items()}
    pred_tags = [id2label.get(p, 'O') for p in all_preds]
    gold_tags = [id2label.get(g, 'O') for g in all_labels]

    valid = [g in ['B', 'I', 'O'] for g in gold_tags]
    filtered_preds = [p for p, v in zip(pred_tags, valid) if v]
    filtered_golds = [g for g, v in zip(gold_tags, valid) if v]

    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_golds, filtered_preds, labels=['B', 'I'], average='micro', zero_division=0)

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, precision, recall, f1
