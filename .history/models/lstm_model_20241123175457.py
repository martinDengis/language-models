import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
import json
import os
import glob
import time
from config import learning_rate, nepochs, batch_size, hidden_size, num_layers, dropout
from utils import save_and_plot_results

# LSTM Model Definition
class LSTM(nn.Module):
    """
    A Long Short-Term Memory (LSTM) neural network module.

    Args:
        num_emb (int): The size of the input vocabulary.
        output_size (int): The size of the output layer.
        num_layers (int, optional): The number of LSTM layers. Default is 1.
        hidden_size (int, optional): The number of features in the hidden state. Default is 128.

    Attributes:
        embedding (nn.Embedding): Embedding layer that converts input indices to dense vectors.
        lstm (nn.LSTM): LSTM layer(s) for processing sequences.
        fc_out (nn.Linear): Fully connected layer that maps LSTM outputs to the desired output size.

    Methods:
        forward(input_seq, hidden_in, mem_in):
            Performs a forward pass of the LSTM network.

            Args:
                input_seq (Tensor): Input sequence tensor of shape (batch_size, seq_length).
                hidden_in (Tensor): Initial hidden state tensor of shape (num_layers, batch_size, hidden_size).
                mem_in (Tensor): Initial cell state tensor of shape (num_layers, batch_size, hidden_size).

            Returns:
                Tuple[Tensor, Tensor, Tensor]: Output tensor of shape (batch_size, seq_length, output_size),
                                               hidden state tensor of shape (num_layers, batch_size, hidden_size),
                                               and cell state tensor of shape (num_layers, batch_size, hidden_size).
    """
    def __init__(self, num_emb, output_size, num_layers=1, hidden_size=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=0.5)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_in, mem_in):
        input_embs = self.embedding(input_seq)
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
        return self.fc_out(output), hidden_out, mem_out
    
def initialize_lstm_model(vocab, num_layers, hidden_size, checkpoint_dir=None, device=torch.device("cpu")):
    """
    Initializes an LSTM model, optimizer, and related training state.

    Args:
        vocab (Vocab): Vocabulary object to determine input/output size.
        num_layers (int): Number of LSTM layers.
        hidden_size (int): Hidden size for the LSTM layers.
        checkpoint_dir (str, optional): Path to the checkpoint directory.
        device (torch.device): Device to load the model on.

    Returns:
        model (nn.Module): Initialized LSTM model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        start_epoch (int): Epoch to resume training from (or 0 if starting fresh).
        train_losses (list): List of training losses from checkpoint (or empty list).
        test_losses (list): List of test losses from checkpoint (or empty list).
        perplexities (list): List of perplexities from checkpoint (or empty list).
        loss_fn (nn.Module): Loss function for training.
    """
    if checkpoint_dir:
        checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pth")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
            checkpoint_path = checkpoint_files[-1]
            print(f"Checkpoint found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path)
            model = LSTM(num_emb=len(vocab), output_size=len(vocab),
                         num_layers=num_layers, hidden_size=hidden_size).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer = optim.Adam(model.parameters(), lr=checkpoint.get('learning_rate', 1e-3))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            perplexities = checkpoint['perplexities']

            print(f"Model loaded. Resuming from epoch {start_epoch}.")
        else:
            print("No checkpoint found, initializing a new LSTM model.")
            model, optimizer, start_epoch, train_losses, test_losses, perplexities = initialize_new_lstm(vocab, num_layers, hidden_size, device)
    else:
        model, optimizer, start_epoch, train_losses, test_losses, perplexities = initialize_new_lstm(vocab, num_layers, hidden_size, device)

    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, start_epoch, train_losses, test_losses, perplexities, loss_fn

def initialize_new_lstm(vocab, num_layers, hidden_size, device):
    model = LSTM(num_emb=len(vocab), output_size=len(vocab), num_layers=num_layers, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    train_losses = []
    test_losses = []
    perplexities = []
    print("Initialized new LSTM model.")
    return model, optimizer, start_epoch, train_losses, test_losses, perplexities

def train_lstm(train_loader, test_loader, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, start_epoch, train_losses, test_losses, perplexities, loss_fn = initialize_lstm_model(vocab, device)

    start_time = time.time()

    epoch_bar = trange(start_epoch, nepochs, desc="Training Progress")

    for epoch in epoch_bar:
        model.train()
        train_loss = 0
        train_steps = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nepochs}", leave=False, ncols=100, mininterval=1.0)

        for sequences, targets in batch_bar:
            sequences = sequences.to(device)
            targets = targets.to(device)
            bs = sequences.shape[0]
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
            memory = torch.zeros(num_layers, bs, hidden_size, device=device)

            output, hidden, memory = model(sequences, hidden, memory)
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
                memory = torch.zeros(num_layers, bs, hidden_size, device=device)

                output, hidden, memory = model(sequences, hidden, memory)
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
            print(f"\nSample text generation: {sample_text}\n")

        if (epoch + 1) % 5 == 0 or epoch == nepochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'perplexities': perplexities
            }, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")

    training_time = time.time() - start_time
    save_and_plot_results("LSTM", train_losses, test_losses, perplexities, training_time)
    return train_losses, test_losses, perplexities

def save_results(model_name, train_losses, test_losses, perplexities):
    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "perplexities": perplexities
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(results, f)

def generate_text(model, vocab, tokenizer, seed_text, max_length=20):
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer(seed_text)
    current_sequence = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

    hidden = torch.zeros(num_layers, 1, hidden_size, device=device)
    memory = torch.zeros(num_layers, 1, hidden_size, device=device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            input_sequence = current_sequence[-1:].unsqueeze(0).to(device)
            output, hidden, memory = model(input_sequence, hidden, memory)
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

# LSTM Model (One-Hot Encoding)
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
    
class LSTMWord2Vec(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, w2v_model, vocab, dropout=0.2):
        super(LSTMWord2Vec, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Create embedding matrix from Word2Vec
        embedding_dim = w2v_model.vector_size
        embedding_matrix = self.create_embedding_matrix(w2v_model, vocab, embedding_dim)

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def create_embedding_matrix(self, w2v_model, vocab, embedding_dim):
        embedding_matrix = torch.zeros(len(vocab), embedding_dim)
        for token, idx in vocab.get_stoi().items():
            if token in w2v_model.wv:
                embedding_matrix[idx] = torch.tensor(w2v_model.wv[token])
        return embedding_matrix

    def forward(self, x, hidden, memory):
        embedded = self.embedding(x)
        output, (hidden, memory) = self.lstm(embedded, (hidden, memory))
        output = self.fc(output)
        return output, hidden, memory