import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_emb, output_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, memory):
        x = self.embedding(x)
        output, (hidden, memory) = self.lstm(x, (hidden, memory))
        return self.fc(output), hidden, memory

class GRU(nn.Module):
    def __init__(self, num_emb, output_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.gru(x, hidden)
        return self.fc(output), hidden
