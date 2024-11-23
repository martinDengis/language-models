import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
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

def train_gru(train_loader, test_loader, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRU(num_emb=len(vocab), output_size=len(vocab)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, test_losses, perplexities = [], [], []

    start_time = time.time()

    with tqdm(total=len(train_loader) * nepochs, desc="Training GRU", unit="batch") as pbar:
        for epoch in range(nepochs):
            model.train()
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                hidden = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device)
                output, hidden = model(sequences, hidden)
                output = output.view(-1, len(vocab))
                targets = targets.view(-1)
                loss = loss_fn(output, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                pbar.update(1)

            model.eval()
            with torch.no_grad():
                for sequences, targets in test_loader:
                    sequences, targets = sequences.to(device), targets.to(device)
                    hidden = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device)
                    output, hidden = model(sequences, hidden)
                    output = output.view(-1, len(vocab))
                    targets = targets.view(-1)
                    loss = loss_fn(output, targets)
                    test_losses.append(loss.item())
                    perplexities.append(torch.exp(loss).item())

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