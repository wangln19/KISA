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
        # representation.shape torch.Size([32, 300])

        prob = self.dropout(self.fc(representation))

        return representation, prob


class FusionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, finetune=False):
        super(FusionClassifier, self).__init__()
        self.finetune = finetune
        self.lstm1 = nn.LSTM(input_size, hidden_size, layer_num)
        self.lstm2 = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(2 * hidden_size, 2)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, x_length):
        x1 = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x1, hidden = self.lstm1(x1)
        output, _ = pad_packed_sequence(x1)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = output.view(seq_len, batch_size, -1)

        representation1 = []
        for i, length in enumerate(x_length):
            representation1.append(output[length - 1, i, :])
        representation1 = torch.stack(representation1, dim=0)

        x2 = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x2, hidden = self.lstm2(x2)
        output, _ = pad_packed_sequence(x2)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = output.view(seq_len, batch_size, -1)

        representation2 = []
        for i, length in enumerate(x_length):
            representation2.append(output[length - 1, i, :])
        representation2 = torch.stack(representation2, dim=0)

        representation = torch.cat((representation1, representation2), 1)
        prob = self.dropout(self.fc(representation))

        return representation, prob
    

class ShortCut_LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super(ShortCut_LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size + 24 + 2, 2)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, x_length, shortcut_var):
        x = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x, hidden = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = output.view(seq_len, batch_size, -1)
        # print('output.size()',output.size()) torch.Size([20, 32, 300])   

        representation = []
        for i, length in enumerate(x_length):
            representation.append(output[length - 1, i, :])
        representation = torch.stack(representation, 0)
        sc_representation = torch.cat((representation, shortcut_var), dim=1)
        # print('sc_representation.shape',sc_representation.shape)# torch.Size([32, 326])

        prob = self.dropout(self.fc(sc_representation))

        return representation, prob