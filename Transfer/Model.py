"""
Models
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTMTransfer(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super(LSTMTransfer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.8)
        self.src_weight = nn.Parameter(torch.randn(hidden_size, 128))
        self.tgt_weight = nn.Parameter(torch.randn(hidden_size, 128))

        # self.src_weight = nn.Linear(hidden_size, 128, bias=True)
        # self.tgt_weight = nn.Linear(hidden_size, 128, bias=True)

    def forward(self, x, x_length):
        x = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x, hidden = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = output.view(seq_len, batch_size, -1)

        representation = []
        for i, length in enumerate(x_length):
            representation.append(output[length - 1, i, :])
        representation = torch.stack(representation, dim=0)

        return representation, self.dropout(self.fc(representation))


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, finetune=False):
        super(LSTMClassifier, self).__init__()
        self.finetune = finetune
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, x_length):
        x = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x, hidden = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = output.view(seq_len, batch_size, -1)

        representation = []
        for i, length in enumerate(x_length):
            representation.append(output[length - 1, i, :])
        representation = torch.stack(representation, dim=0)

        prob = self.dropout(self.fc(representation))

        return representation, prob