import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
import os
from config import learning_rate, nepochs, batch_size, hidden_size, num_layers, dropout
from utils import save_and_plot_results

# GRU Model Definition
class GRU(nn.Module):
    def __init__(self, num_emb, output_size, num_layers, hidden_size, dropout=0.2):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(num_emb, hidden_size)

        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Embedding shape: batch_size x sequence_length x hidden_size
        embedding = self.embedding(x)

        # GRU output shape: batch_size x sequence_length x hidden_size
        output, hidden = self.gru(embedding, hidden)

        # Reshape output for linear layer
        output = output.contiguous().view(-1, self.hidden_size)

        # Linear layer shape: batch_size*sequence_length x output_size
        output = self.fc(output)

        return output, hidden

def initialize_gru_model(vocab, num_layers, hidden_size, device=torch.device("cpu")):
    """
    Initializes a GRU model and optimizer.

    Args:
        vocab (Vocab): Vocabulary object to determine input/output size.
        num_layers (int): Number of GRU layers.
        hidden_size (int): Hidden size for the GRU layers.
        device (torch.device): Device to load the model on.

    Returns:
        model (nn.Module): Initialized GRU model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        loss_fn (nn.Module): Loss function for training.
    """
    print("Initializing GRU model...")
    model = GRU(num_emb=len(vocab), output_size=len(vocab), num_layers=num_layers, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("\nGRU Model Architecture:")
    print(model)
    print(f"\nTotal GRU parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, optimizer, loss_fn


def train_gru(train_loader, test_loader, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, loss_fn = initialize_gru_model(vocab, device)
    train_losses, test_losses, perplexities = [], [], []

    start_time = time.time()

    epoch_bar = trange(nepochs, desc="GRU Training Progress")

    for epoch in epoch_bar:
        model.train()
        train_loss = 0
        train_steps = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nepochs}", leave=False, ncols=100, mininterval=1.0)

        for sequences, targets in batch_bar:
            sequences, targets = sequences.to(device), targets.to(device)
            bs = sequences.shape[0]
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)

            output, hidden = model(sequences, hidden)
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)

            loss = loss_fn(output, targets)
            train_loss += loss.item()
            train_steps += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0
        test_steps = 0

        with torch.no_grad():
            for sequences, targets in tqdm(test_loader, desc="Validation", leave=False):
                sequences = sequences.to(device)
                targets = targets.to(device)
                bs = sequences.shape[0]
                hidden = torch.zeros(num_layers, bs, hidden_size, device=device)

                output, hidden = model(sequences, hidden)
                output = output.view(-1, len(vocab))
                targets = targets.view(-1)

                loss = loss_fn(output, targets)
                test_loss += loss.item()
                test_steps += 1

        avg_test_loss = test_loss / test_steps
        test_losses.append(avg_test_loss)

        perplexity = calculate_perplexity(avg_test_loss)
        perplexities.append(perplexity)

        epoch_bar.set_postfix(train_loss=f"{avg_train_loss:.4f}", test_loss=f"{avg_test_loss:.4f}", perplexity=f"{perplexity:.2f}")

        if (epoch + 1) % 2 == 0:
            sample_text = generate_text(model, vocab, tokenizer, "Le président", max_length=20)
            print(f"\nGRU Sample text generation: {sample_text}\n")

    training_time = time.time() - start_time
    save_and_plot_results("GRU", train_losses, test_losses, perplexities, training_time)
    return train_losses, test_losses, perplexities

def generate_text(model, vocab, tokenizer, seed_text, max_length=20):
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer(seed_text)
    current_sequence = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

    hidden = torch.zeros(num_layers, 1, hidden_size, device=device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            input_sequence = current_sequence[-1:].unsqueeze(0).to(device)
            output, hidden = model(input_sequence, hidden)
            probs = nn.functional.softmax(output.squeeze(), dim=-1)
            next_token_idx = torch.multinomial(probs, 1).item()
            current_sequence = torch.cat([current_sequence, torch.tensor([next_token_idx])])
            for token, idx in vocab.get_stoi().items():
                if idx == next_token_idx:
                    generated_tokens.append(token)
                    break

    return seed_text + ' ' + ' '.join(generated_tokens)

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()

# GRU Model (One-Hot Encoding)
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
    

class GRUWord2Vec(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, w2v_model, vocab, dropout=0.2):
        super(GRUWord2Vec, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Create embedding matrix from Word2Vec
        embedding_dim = w2v_model.vector_size
        embedding_matrix = self.create_embedding_matrix(w2v_model, vocab, embedding_dim)

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def create_embedding_matrix(self, w2v_model, vocab, embedding_dim):
        embedding_matrix = torch.zeros(len(vocab), embedding_dim)
        for token, idx in vocab.get_stoi().items():
            if token in w2v_model.wv:
                embedding_matrix[idx] = torch.tensor(w2v_model.wv[token])
        return embedding_matrix

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden