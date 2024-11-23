import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import time
import json
import os
from config import learning_rate, nepochs, batch_size, hidden_size, num_layers, dropout
from utils import save_and_plot_results

def train_model(model, train_loader, test_loader, vocab, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, test_losses, perplexities = [], [], []

    start_time = time.time()

    with tqdm(total=len(train_loader) * nepochs, desc=f"Training {model_name}", unit="batch") as pbar:
        for epoch in range(nepochs):
            model.train()
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                hidden = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device)
                memory = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device) if hasattr(model, 'lstm') else None
                if hasattr(model, 'lstm'):
                    output, hidden, memory = model(sequences, hidden, memory)
                else:
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
                    memory = torch.zeros(num_layers, sequences.size(0), hidden_size, device=device) if hasattr(model, 'lstm') else None
                    if hasattr(model, 'lstm'):
                        output, hidden, memory = model(sequences, hidden, memory)
                    else:
                        output, hidden = model(sequences, hidden)
                    output = output.view(-1, len(vocab))
                    targets = targets.view(-1)
                    loss = loss_fn(output, targets)
                    test_losses.append(loss.item())
                    perplexities.append(torch.exp(loss).item())

    save_results(model_name, train_losses, test_losses, perplexities)

    # Training Summary and Visualization
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    # Plot perplexity
    plt.subplot(1, 2, 2)
    plt.plot(perplexities, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Model Perplexity')
    plt.tight_layout()

    # Save the plots
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_name}_training_summary.png")
    plt.show()

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

def plot_results(lstm_results, gru_results):
    lstm_train_losses, lstm_test_losses, lstm_perplexities = lstm_results
    gru_train_losses, gru_test_losses, gru_perplexities = gru_results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM Train Loss')
    plt.plot(gru_train_losses, label='GRU Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(lstm_test_losses, label='LSTM Test Loss')
    plt.plot(gru_test_losses, label='GRU Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(lstm_perplexities, label='LSTM Perplexity')
    plt.plot(gru_perplexities, label='GRU Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison')
    plt.legend()
    plt.show()

def generate_text(model, vocab, tokenizer, seed_text, max_length=20):
    """
    Generate text using the trained model (LSTM or GRU)

    Args:
        model: The trained model (LSTM or GRU)
        vocab: Vocabulary object
        tokenizer: Tokenizer function
        seed_text: Initial text to start generation
        max_length: Maximum number of tokens to generate

    Returns:
        str: The generated text
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize seed text
    tokens = tokenizer(seed_text)
    current_sequence = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

    # Initialize hidden states
    hidden = torch.zeros(num_layers, 1, hidden_size, device=device)

    # Initialize memory cell only for LSTM
    memory = torch.zeros(num_layers, 1, hidden_size, device=device) if isinstance(model, LSTM) else None

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_sequence = current_sequence[-1:].unsqueeze(0).to(device)

            # Get model prediction based on model type
            if isinstance(model, LSTM):
                output, hidden, memory = model(input_sequence, hidden, memory)
            else:  # GRU case
                output, hidden = model(input_sequence, hidden)

            # Get probabilities and sample next token
            probs = F.softmax(output.squeeze(), dim=-1)
            next_token_idx = torch.multinomial(probs, 1).item()

            # Append to generated sequence
            current_sequence = torch.cat([current_sequence, torch.tensor([next_token_idx])])

            # Get the actual token
            for token, idx in vocab.get_stoi().items():
                if idx == next_token_idx:
                    generated_tokens.append(token)
                    break

    return seed_text + ' ' + ' '.join(generated_tokens)

def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss)).item()