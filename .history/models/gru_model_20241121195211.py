import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
from config import learning_rate, nepochs, batch_size, hidden_size, num_layers, dropout
from utils import save_and_plot_results

class GRU(nn.Module):
    def __init__(self, num_emb, output_size, num_layers=num_layers, hidden_size=hidden_size):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_in):
        input_embs = self.embedding(input_seq)
        output, hidden_out = self.gru(input_embs, hidden_in)
        return self.fc_out(output), hidden_out

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