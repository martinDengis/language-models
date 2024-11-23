import torch
from gensim.models import Word2Vec
from training import train_model
from models import GRUWord2Vec
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from config import hidden_size, num_layers, dropout

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
    """
    Trains a Word2Vec model on the tokenized corpus.

    Args:
        tokenized_corpus (list of list of str): Tokenized corpus.
        embedding_dim (int): Size of the Word2Vec embeddings.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    print("Training Word2Vec embeddings...")
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    print("Word2Vec training complete.")
    return w2v_model

def experiment_1(train_loader, test_loader, vocab):
    """
    Compares LSTM and GRU performance on the same dataset.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        vocab (Vocab): Vocabulary object for token mapping.
    """
    # Initialize models
    lstm = LSTM(num_emb=len(vocab), output_size=len(vocab), hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout)
    gru = GRU(num_emb=len(vocab), output_size=len(vocab), hidden_size=hidden_size,
              num_layers=num_layers, dropout=dropout)

    # Train models
    print("Training LSTM...")
    lstm_results = train_model(lstm, train_loader, test_loader, vocab, "LSTM")
    
    print("Training GRU...")
    gru_results = train_model(gru, train_loader, test_loader, vocab, "GRU")

    # Compare results
    compare_models(lstm_results, gru_results, "LSTM", "GRU")


def experiment_2(train_loader, test_loader, vocab):
    """
    Evaluates the impact of different hyperparameters (e.g., hidden size) on LSTM performance.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        vocab (Vocab): Vocabulary object for token mapping.
    """
    hidden_sizes = [64, 128, 256]
    for hs in hidden_sizes:
        print(f"Training LSTM with hidden size = {hs}...")
        lstm = LSTM(num_emb=len(vocab), output_size=len(vocab), hidden_size=hs,
                    num_layers=num_layers, dropout=dropout)
        train_losses, test_losses, perplexities = train_model(lstm, train_loader, test_loader, vocab,
                                                              f"LSTM_hidden_{hs}")
        print(f"Results for hidden size {hs}:")
        print(f"Train Losses: {train_losses}")
        print(f"Test Losses: {test_losses}")
        print(f"Perplexities: {perplexities}")
