import torch
import torch.nn as nn
from torch import optim
from gensim.models import Word2Vec
import os

class LSTMOneHot(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMOneHot, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, memory):
        x_one_hot = torch.zeros(x.size(0), x.size(1), self.vocab_size, device=x.device)
        x_one_hot.scatter_(2, x.unsqueeze(-1), 1)
        output, (hidden, memory) = self.lstm(x_one_hot, (hidden, memory))
        output = self.fc(output)
        return output, hidden, memory

class GRUOneHot(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.2):
        super(GRUOneHot, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.gru = nn.GRU(vocab_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x_one_hot = torch.zeros(x.size(0), x.size(1), self.vocab_size, device=x.device)
        x_one_hot.scatter_(2, x.unsqueeze(-1), 1)
        output, hidden = self.gru(x_one_hot, hidden)
        output = self.fc(output)
        return output, hidden

class LSTMWord2Vec(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, w2v_model, vocab, dropout=0.2):
        super(LSTMWord2Vec, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        embedding_dim = w2v_model.vector_size
        embedding_matrix = self.create_embedding_matrix(w2v_model, vocab, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def create_embedding_matrix(self, w2v_model, vocab, embedding_dim):
        embedding_matrix = torch.zeros(len(vocab), embedding_dim)
        for token, idx in vocab.items():
            if token in w2v_model.wv:
                embedding_matrix[idx] = torch.tensor(w2v_model.wv[token])
        return embedding_matrix

    def forward(self, x, hidden, memory):
        embedded = self.embedding(x)
        output, (hidden, memory) = self.lstm(embedded, (hidden, memory))
        output = self.fc(output)
        return output, hidden, memory

class GRUWord2Vec(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, w2v_model, vocab, dropout=0.2):
        super(GRUWord2Vec, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        embedding_dim = w2v_model.vector_size
        embedding_matrix = self.create_embedding_matrix(w2v_model, vocab, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def create_embedding_matrix(self, w2v_model, vocab, embedding_dim):
        embedding_matrix = torch.zeros(len(vocab), embedding_dim)
        for token, idx in vocab.items():
            if token in w2v_model.wv:
                embedding_matrix[idx] = torch.tensor(w2v_model.wv[token])
        return embedding_matrix

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden

def train_word2vec(tokenized_corpus, embedding_dim=100):
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    return w2v_model

def run_experiments(train_loader, test_loader, vocab):
    hidden_size = 128
    num_layers = 2
    dropout = 0.2

    lstm_onehot = LSTMOneHot(len(vocab), hidden_size, num_layers, dropout).to(device)
    gru_onehot = GRUOneHot(len(vocab), hidden_size, num_layers, dropout).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_lstm = optim.Adam(lstm_onehot.parameters(), lr=1e-4)
    optimizer_gru = optim.Adam(gru_onehot.parameters(), lr=1e-4)

    for epoch in range(10):
        lstm_onehot.train()
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            hidden = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device)
            memory = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device)
            output, hidden, memory = lstm_onehot(sequences, hidden, memory)
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)
            loss = loss_fn(output, targets)
            optimizer_lstm.zero_grad()
            loss.backward()
            optimizer_lstm.step()

    for epoch in range(10):
        gru_onehot.train()
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            hidden = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device)
            output, hidden = gru_onehot(sequences, hidden)
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)
            loss = loss_fn(output, targets)
            optimizer_gru.zero_grad()
            loss.backward()
            optimizer_gru.step()


def train_word2vec(tokenized_corpus, embedding_dim=100):
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    return w2v_model

def experiment_1(train_loader, test_loader, vocab):
    # Implémentez ici l'expérience 1
    print("Running Experiment 1...")
    # Exemple : Entraînez un modèle LSTM avec One-Hot Encoding
    model = LSTMOneHot(vocab_size=len(vocab), hidden_size=hidden_size, num_layers=num_layers)
    # Utilisez les fonctions d'entraînement et de validation existantes
    train_model(model, train_loader, test_loader, vocab, "LSTM_OneHot")

def experiment_2(train_loader, test_loader, vocab):
    # Implémentez ici l'expérience 2
    print("Running Experiment 2...")
    # Exemple : Entraînez un modèle GRU avec Word2Vec
    tokenized_corpus = [list(vocab.keys())]  # Remplacez par votre corpus tokenisé
    w2v_model = train_word2vec(tokenized_corpus)
    model = GRUWord2Vec(vocab_size=len(vocab), hidden_size=hidden_size, num_layers=num_layers, w2v_model=w2v_model, vocab=vocab)
    # Utilisez les fonctions d'entraînement et de validation existantes
    train_model(model, train_loader, test_loader, vocab, "GRU_Word2Vec")