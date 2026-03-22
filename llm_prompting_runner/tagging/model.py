import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasedPhraseTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, padding_idx=0):
        super(BiasedPhraseTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, lengths):
        embeds = self.embedding(input_ids)
        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.hidden2label(output)
        return logits
