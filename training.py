import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import time
import json
import os
from config import learning_rate, nepochs, batch_size, hidden_size, num_layers
from utils import save_and_plot_results, calculate_perplexity

def train_model(model, train_loader, test_loader, optimizer, loss_fn, num_epochs, device, model_name, vocab):
    """
    Trains a given model on the provided training data and evaluates it on test data.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (nn.Module): Loss function for training.
        num_epochs (int): Number of epochs to train.
        device (torch.device): The device to use (CPU or GPU).
        model_name (str): Name of the model, used for saving results.
        vocab (Vocab): Vocabulary object for token mapping.

    Returns:
        tuple: (train_losses, test_losses, perplexities)
    """
    train_losses, test_losses, perplexities = [], [], []

    # Single progress bar for the entire training process
    with tqdm(total=num_epochs, desc=f"Training {model_name}", leave=True) as progress_bar:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            total_train_steps = 0

            for sequences, targets in train_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                hidden = torch.zeros(model.num_layers, sequences.size(0), model.hidden_size, device=device)
                memory = hidden.clone() if hasattr(model, 'lstm') else None

                if memory is not None:
                    output, hidden, memory = model(sequences, hidden, memory)
                else:
                    output, hidden = model(sequences, hidden)

                output = output.view(-1, len(vocab))
                targets = targets.view(-1)
                loss = loss_fn(output, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_steps += 1

            avg_train_loss = total_train_loss / total_train_steps
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            total_test_loss = 0
            total_test_steps = 0

            with torch.no_grad():
                for sequences, targets in test_loader:
                    sequences, targets = sequences.to(device), targets.to(device)
                    hidden = torch.zeros(model.num_layers, sequences.size(0), model.hidden_size, device=device)
                    memory = hidden.clone() if hasattr(model, 'lstm') else None

                    if memory is not None:
                        output, hidden, memory = model(sequences, hidden, memory)
                    else:
                        output, hidden = model(sequences, hidden)

                    output = output.view(-1, len(vocab))
                    targets = targets.view(-1)
                    loss = loss_fn(output, targets)

                    total_test_loss += loss.item()
                    total_test_steps += 1

            avg_test_loss = total_test_loss / total_test_steps
            test_losses.append(avg_test_loss)

            perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
            perplexities.append(perplexity)

            # Update progress bar with epoch metrics
            progress_bar.set_postfix(
                train_loss=f"{avg_train_loss:.4f}",
                test_loss=f"{avg_test_loss:.4f}",
                perplexity=f"{perplexity:.2f}"
            )
            progress_bar.update(1)

    return train_losses, test_losses, perplexities

def save_results(model_name, train_losses, test_losses, perplexities):
    """
    Saves training results to a JSON file.

    Args:
        model_name (str): Name of the model.
        train_losses (list): Training losses.
        test_losses (list): Testing losses.
        perplexities (list): Perplexity values.
    """
    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "perplexities": perplexities
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(results, f)

def generate_text(model, vocab, tokenizer, seed_text, max_length=20):
    """
    Generates text using a trained model.

    Args:
        model (nn.Module): Trained PyTorch model.
        vocab (Vocab): Vocabulary object for token mapping.
        tokenizer (callable): Function to tokenize input text.
        seed_text (str): Initial text to seed the generator.
        max_length (int): Maximum length of generated text.

    Returns:
        str: Generated text.
    """
    model.eval()
    device = next(model.parameters()).device

    tokens = tokenizer(seed_text)
    current_sequence = torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0).to(device)

    hidden = torch.zeros(num_layers, 1, hidden_size, device=device)
    memory = hidden.clone() if hasattr(model, 'lstm') else None

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            if memory is not None:
                output, hidden, memory = model(current_sequence, hidden, memory)
            else:
                output, hidden = model(current_sequence, hidden)

            probs = torch.nn.functional.softmax(output[:, -1, :], dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            current_sequence = torch.cat([current_sequence, torch.tensor([[next_token_idx]], device=device)], dim=1)

            # Convert token index back to token
            for token, idx in vocab.get_stoi().items():
                if idx == next_token_idx:
                    generated_tokens.append(token)
                    break

    return seed_text + ' ' + ' '.join(generated_tokens)

    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss)).item()