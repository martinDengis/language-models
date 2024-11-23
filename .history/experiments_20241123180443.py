import time
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from training import train_model
from models import GRUWord2Vec, LSTMOneHot, GRUOneHot
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from utils import plot_training_summary, compare_model_results
from config import hidden_size, num_layers, dropout, nepochs, learning_rate
from setup import setup_environment

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

# Train Word2Vec
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

# Experiment 1: Compare LSTM and GRU
def experiment_1(train_loader, test_loader, vocab, device):
    """
    Compares LSTM and GRU performance on the same dataset.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        vocab (Vocab): Vocabulary object for token mapping.
        device (torch.device): Device to load models on.
    """
    print("Running Experiment 1: LSTM vs GRU with One-Hot Encoding.")

    # Initialize models
    lstm_model = LSTMOneHot(len(vocab), hidden_size, num_layers, dropout).to(device)
    gru_model = GRUOneHot(len(vocab), hidden_size, num_layers, dropout).to(device)

    # Optimizers and loss functions
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train models
    print("Training LSTM...")
    lstm_results = train_model(
    model=lstm_model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=lstm_optimizer,
    loss_fn=loss_fn,
    num_epochs=nepochs,
    device=device,
    model_name="LSTM",
    vocab=vocab
)
    lstm_training_time = time.time()

    print("Training GRU...")
    gru_results = train_model(
    model=gru_model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=lstm_optimizer,
    loss_fn=loss_fn,
    num_epochs=nepochs,
    device=device,
    model_name="GRU",
    vocab=vocab
)
    gru_training_time = time.time()

    # Compare and visualize results
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    gru_params = sum(p.numel() for p in gru_model.parameters())

    compare_model_results(
        lstm_results,
        gru_results,
        lstm_params,
        gru_params,
        lstm_training_time,
        gru_training_time
    )

# Experiment 2: GRU with Word2Vec Embeddings
def experiment_2(train_loader, test_loader, vocab, tokenized_corpus, device):
    """
    Trains and evaluates a GRU model using Word2Vec embeddings.

    Args:
        train_loader (DataLoader): Training DataLoader.
        test_loader (DataLoader): Testing DataLoader.
        vocab (Vocab): Vocabulary object for token mapping.
        tokenized_corpus (list of list of str): Tokenized corpus for Word2Vec training.
        device (torch.device): The device to use (CPU or GPU).
    """
    print("Running Experiment 2: GRU with Word2Vec embeddings.")

    # Train Word2Vec model
    embedding_dim = 100
    w2v_model = train_word2vec(tokenized_corpus, embedding_dim)

    # Initialize GRU model with Word2Vec embeddings
    model = GRUWord2Vec(len(vocab), hidden_size, num_layers, w2v_model, vocab, dropout).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train and evaluate the model
    train_model(model, train_loader, test_loader, optimizer, loss_fn, nepochs, device, "GRU_Word2Vec", vocab)

