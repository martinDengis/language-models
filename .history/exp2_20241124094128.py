
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from tqdm import trange, tqdm
import time
import os
import json
from datetime import datetime
import spacy
from gensim.models import Word2Vec
import warnings
from setup import setup_environment
from config import learning_rate, nepochs, batch_size, max_len, hidden_size, num_layers
from lstm import create_experiment_dir, hyperparams, corpus_path, vocab, train_loader, test_loader, tokenizer, lstm_train_losses, lstm_test_losses, perplexities, lstm_training_time, total_params
warnings.filterwarnings('ignore')

# Set up the environment and get the device
device = setup_environment()
# Create experiment directory
exp_dir = create_experiment_dir(prefix="exp2{hidden_size}_l{num_layers}")

# # **Experiment 2**: Investigate the performance using 1-hot encoding and Word2Vec as input.

# ## One-hot Encoder
class LSTMOneHot(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMOneHot, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # One-hot input will be vocab_size dimensional
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, memory):
        # Convert input to one-hot
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

        # One-hot input will be vocab_size dimensional
        self.gru = nn.GRU(vocab_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # Convert input to one-hot
        x_one_hot = torch.zeros(x.size(0), x.size(1), self.vocab_size, device=x.device)
        x_one_hot.scatter_(2, x.unsqueeze(-1), 1)

        output, hidden = self.gru(x_one_hot, hidden)
        output = self.fc(output)
        return output, hidden


# ## Word2Vec Encoder

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


# ## Training Functions

# Train Word2Vec model
def train_word2vec(tokenized_corpus, embedding_dim=100):
    """Train Word2Vec model on the corpus"""
    w2v_model = Word2Vec(sentences=tokenized_corpus,
                        vector_size=embedding_dim,
                        window=5,
                        min_count=2,
                        workers=4)
    return w2v_model


# Training function for all models
def train_model(model, train_loader, test_loader, optimizer, loss_fn, num_epochs, device, model_type):
    """Generic training function for both types of models"""
    train_losses = []
    test_losses = []
    perplexities = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            batch_size = sequences.size(0)

            # Initialize hidden (and memory for LSTM)
            hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

            if isinstance(model, (LSTMOneHot, LSTMWord2Vec)):
                memory = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                output, _, _ = model(sequences, hidden, memory)
            else:  # GRU models
                output, _ = model(sequences, hidden)

            loss = loss_fn(output.view(-1, model.vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        test_loss = 0
        test_steps = 0

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                batch_size = sequences.size(0)

                hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

                if isinstance(model, (LSTMOneHot, LSTMWord2Vec)):
                    memory = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    output, _, _ = model(sequences, hidden, memory)
                else:  # GRU models
                    output, _ = model(sequences, hidden)

                loss = loss_fn(output.view(-1, model.vocab_size), targets.view(-1))
                test_loss += loss.item()
                test_steps += 1

        avg_test_loss = test_loss / test_steps
        test_losses.append(avg_test_loss)

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
        perplexities.append(perplexity)

        print(f'Epoch {epoch+1}/{num_epochs} | {model_type}')
        print(f'Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Perplexity: {perplexity:.2f}')

    return train_losses, test_losses, perplexities


# ## Run Experiments

# Initialize and train all models
def run_experiments(train_loader, test_loader, vocab, device, config):
    """Run experiments with different models and embeddings"""
    results = {}
    drive_path = '/content/drive/MyDrive/Colab Notebooks/web-text-analytics/'

    # Create directory if it doesn't exist
    if not os.path.exists(drive_path):
        os.makedirs(drive_path)

    # Prepare Word2Vec model
    w2v_model = train_word2vec(config['tokenized_corpus'], config['embedding_dim'])

    # Initialize models
    models = {
        'LSTM_OneHot': LSTMOneHot(len(vocab), config['hidden_size'], config['num_layers']),
        'GRU_OneHot': GRUOneHot(len(vocab), config['hidden_size'], config['num_layers']),
        'LSTM_W2V': LSTMWord2Vec(len(vocab), config['hidden_size'], config['num_layers'], w2v_model, vocab),
        'GRU_W2V': GRUWord2Vec(len(vocab), config['hidden_size'], config['num_layers'], w2v_model, vocab)
    }

    # Train each model
    for name, model in models.items():
        model_path = os.path.join(drive_path, f'model_{name}.pt')

        if os.path.exists(model_path):
            print(f"\nLoading {name} from {model_path}")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            results[name] = {
                'train_losses': checkpoint['train_losses'],
                'test_losses': checkpoint['test_losses'],
                'perplexities': checkpoint['perplexities']
            }
        else:
            print(f"\nTraining {name}")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            loss_fn = nn.CrossEntropyLoss()

            train_losses, test_losses, perplexities = train_model(
                model, train_loader, test_loader, optimizer, loss_fn,
                config['num_epochs'], device, name
            )

            results[name] = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'perplexities': perplexities
            }

            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'perplexities': perplexities
            }, model_path)

    return results

# Visualization of results
def plot_comparison(results):
    """Plot comparison of all models"""
    plt.figure(figsize=(15, 10))

    # Plot training losses
    plt.subplot(2, 2, 1)
    for name, metrics in results.items():
        plt.plot(metrics['train_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()

    # Plot validation losses
    plt.subplot(2, 2, 2)
    for name, metrics in results.items():
        plt.plot(metrics['test_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()

    # Plot perplexities
    plt.subplot(2, 2, 3)
    for name, metrics in results.items():
        plt.plot(metrics['perplexities'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison')
    plt.legend()

    # Summary metrics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = "Final Metrics Summary\n\n"
    for name, metrics in results.items():
        summary_text += f"{name}:\n"
        summary_text += f"Final Train Loss: {metrics['train_losses'][-1]:.4f}\n"
        summary_text += f"Final Valid Loss: {metrics['test_losses'][-1]:.4f}\n"
        summary_text += f"Final Perplexity: {metrics['perplexities'][-1]:.2f}\n\n"
    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


# Step 1: Prepare tokenized corpus using existing structure
print("Reading and tokenizing corpus...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

# Use the same tokenizer and structure as before
def yield_tokens(text):
    yield tokenizer(text)

tokenized_corpus = list(yield_tokens(corpus))  # Convert generator to list for Word2Vec

# Step 2: Configuration using existing parameters
config = {
    'hidden_size': hidden_size,      # from existing setup
    'num_layers': num_layers,        # from existing setup
    'embedding_dim': 100,            # Word2Vec embedding dimension
    'learning_rate': learning_rate,  # from existing setup
    'num_epochs': nepochs,           # from existing setup
    'tokenized_corpus': tokenized_corpus
}

# Print configuration for verification
print("\nConfiguration:")
print(f"Hidden Size: {config['hidden_size']}")
print(f"Number of Layers: {config['num_layers']}")
print(f"Embedding Dimension: {config['embedding_dim']}")
print(f"Learning Rate: {config['learning_rate']}")
print(f"Number of Epochs: {config['num_epochs']}")
print(f"Vocabulary Size: {len(vocab)}")
print(f"Device: {device}")

# Step 3: Run experiments
print("\nStarting experiments with 4 models:")
print("1. LSTM with One-Hot Encoding")
print("2. GRU with One-Hot Encoding")
print("3. LSTM with Word2Vec")
print("4. GRU with Word2Vec")

results = run_experiments(train_loader, test_loader, vocab, device, config)

# Step 4: Plot and save results
plot_comparison(results)

# Print final comparison
print("\nFinal Results Summary:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"Final Training Loss: {metrics['train_losses'][-1]:.4f}")
    print(f"Final Validation Loss: {metrics['test_losses'][-1]:.4f}")
    print(f"Final Perplexity: {metrics['perplexities'][-1]:.2f}")

