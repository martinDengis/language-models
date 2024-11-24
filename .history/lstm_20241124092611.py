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
import warnings
from setup import setup_environment
from config import learning_rate, nepochs, batch_size, max_len, hidden_size, num_layers
warnings.filterwarnings('ignore')

# Set up the environment and get the device
device = setup_environment()

def create_experiment_dir(base_dir="experiments", prefix="run"):
    """Create a directory for the current experiment with timestamp"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(exp_dir)
    
    return exp_dir

def save_training_plots(exp_dir, train_losses, test_losses, perplexities, sample_generations=None):
    """Save training plots and sample generations"""
    try:
        # Convert lists to numpy arrays explicitly
        train_losses = np.array([float(x) for x in train_losses])
        test_losses = np.array([float(x) for x in test_losses])
        perplexities = np.array([float(x) for x in perplexities])
        
        # Create figure with specific DPI and format
        plt.figure(figsize=(15, 5), dpi=100)
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(np.arange(len(test_losses)), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        
        # Plot perplexity
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(perplexities)), perplexities, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Model Perplexity')
        plt.tight_layout()
        
        # Save the plot with explicit format
        plt.savefig(os.path.join(exp_dir, "training_plots.png"), format='png', bbox_inches='tight')
        plt.close()

        # Save the raw data as CSV
        import pandas as pd
        df = pd.DataFrame({
            'epoch': np.arange(len(train_losses)),
            'train_loss': train_losses,
            'test_loss': test_losses,
            'perplexity': perplexities
        })
        df.to_csv(os.path.join(exp_dir, "training_metrics.csv"), index=False)

        # Save sample generations to a text file if provided
        if sample_generations:
            with open(os.path.join(exp_dir, "sample_generations.txt"), "w", encoding='utf-8') as f:
                for seed, text in sample_generations.items():
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Generated: {text}\n")
                    f.write("-" * 50 + "\n")
    
    except Exception as e:
        print(f"Warning: Could not save plots due to error: {str(e)}")
        print("Saving raw data instead...")
        # Save the raw data as text
        with open(os.path.join(exp_dir, "training_metrics.txt"), "w") as f:
            f.write("Train Losses: " + str(train_losses.tolist()) + "\n")
            f.write("Test Losses: " + str(test_losses.tolist()) + "\n")
            f.write("Perplexities: " + str(perplexities.tolist()) + "\n")
            

# Create experiment directory
exp_dir = create_experiment_dir(prefix=f"lstm_h{hidden_size}_l{num_layers}")

# Store hyperparameters
hyperparams = {
    "learning_rate": learning_rate,
    "nepochs": nepochs,
    "batch_size": batch_size,
    "max_len": max_len,
    "hidden_size": hidden_size,
    "num_layers": num_layers
}

# Adapt corpus path to your drive hierarchy
corpus_path = 'cleaned_lemonde_corpus.txt'

# Custom Dataset for LeMonde corpus
class LemondeDataset(Dataset):
    def __init__(self, file_path, vocab, tokenizer, sequence_length):
        self.sequence_length = sequence_length

        # Read the corpus
        print("Reading corpus from:", file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Corpus length: {len(text)} characters")

        # Tokenize the entire text
        print("Tokenizing text...")
        tokens = tokenizer(text)
        print(f"Number of tokens: {len(tokens)}")

        # Convert tokens to indices
        print("Converting tokens to indices...")
        self.data = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

        # Create sequences and targets
        print("Creating sequences...")
        self.sequences = []
        self.targets = []

        for i in range(0, len(self.data) - sequence_length):
            sequence = self.data[i:i + sequence_length]
            target = self.data[i + 1:i + sequence_length + 1]
            self.sequences.append(sequence)
            self.targets.append(target)

        print(f"Created {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_french_tokenizer():
    """Create and return a French tokenizer function"""
    try:
        nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])

        def tokenize_text(text, chunk_size=900000):
            tokens = []
            chunks = [text[i:i + chunk_size]
                     for i in range(0, len(text), chunk_size)]

            print(f"Processing {len(chunks)} chunks...")

            for chunk in tqdm(chunks):
                doc = nlp(chunk)
                chunk_tokens = [token.text for token in doc]
                tokens.extend(chunk_tokens)

            return tokens

        return tokenize_text

    except OSError:
        print("Installing French language model...")
        os.system('python -m spacy download fr_core_news_sm')
        nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
        return create_french_tokenizer()
    
# Tokenizer and Vocabulary Creation
print("Initializing tokenizer...")
tokenizer = create_french_tokenizer()

# Read corpus for vocab creation
print(f"Reading corpus from {corpus_path}...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

def yield_tokens(text):
    yield tokenizer(text)

# Build vocabulary
print("Building vocabulary...")
vocab = build_vocab_from_iterator(
    yield_tokens(corpus),
    min_freq=2,
    specials=['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
vocab.set_default_index(vocab['<unk>'])
print(f"Vocabulary size: {len(vocab)}")

# Dataset Creation and Splitting
print("Creating dataset...")
dataset = LemondeDataset(corpus_path, vocab, tokenizer, max_len)

print("Splitting dataset...")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# LSTM Model Definition
class LSTM(nn.Module):
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

def generate_text(model, vocab, tokenizer, seed_text, max_length=20):
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
    return torch.exp(torch.tensor(loss)).item()

# Initialize model and optimizer
print("Initializing LSTM model...")
lstm_model = LSTM(num_emb=len(vocab),
             output_size=len(vocab),
             num_layers=num_layers,
             hidden_size=hidden_size).to(device)

lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

# Print model summary
print("\nModel Architecture:")
print(lstm_model)

# Calculate total parameters
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Training Loop
start_time = time.time()
train_losses = []
test_losses = []
perplexities = []

# Initialize the progress bar
epoch_bar = trange(nepochs, desc="Training Progress")

for epoch in epoch_bar:
    # Training phase
    lstm_model.train()
    train_loss = 0
    train_steps = 0

    batch_bar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{nepochs}",
                    leave=False,
                    ncols=100,
                    mininterval=1.0)

    for sequences, targets in batch_bar:
        sequences = sequences.to(device)
        targets = targets.to(device)
        bs = sequences.shape[0]
        hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
        memory = torch.zeros(num_layers, bs, hidden_size, device=device)

        # Forward pass
        output, hidden, memory = lstm_model(sequences, hidden, memory)
        output = output.view(-1, len(vocab))
        targets = targets.view(-1)

        # Calculate loss
        loss = loss_fn(output, targets)
        train_loss += loss.item()
        train_steps += 1

        # Backward pass
        lstm_optimizer.zero_grad()
        loss.backward()
        lstm_optimizer.step()

        # Update progress bar
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / train_steps
    train_losses.append(avg_train_loss)

    # Validation Phase
    lstm_model.eval()
    test_loss = 0
    test_steps = 0

    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc="Validation", leave=False):
            sequences = sequences.to(device)
            targets = targets.to(device)
            bs = sequences.shape[0]
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
            memory = torch.zeros(num_layers, bs, hidden_size, device=device)

            output, hidden, memory = lstm_model(sequences, hidden, memory)
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)

            loss = loss_fn(output, targets)
            test_loss += loss.item()
            test_steps += 1

    avg_test_loss = test_loss / test_steps
    test_losses.append(avg_test_loss)

    # Calculate perplexity
    perplexity = calculate_perplexity(avg_test_loss)
    perplexities.append(perplexity)

    # Update progress bar
    epoch_bar.set_postfix(
        train_loss=f"{avg_train_loss:.4f}",
        test_loss=f"{avg_test_loss:.4f}",
        perplexity=f"{perplexity:.2f}"
    )

    # Generate sample text every few epochs
    if (epoch + 1) % 2 == 0:
        sample_text = generate_text(lstm_model, vocab, tokenizer, "Le président", max_length=20)
        print(f"\nSample text generation: {sample_text}\n")

# Training Summary
training_time = time.time() - start_time

# Generate multiple sample texts with different seeds
sample_seeds = [
    "Le président",
    "La France",
    "Les électeurs",
    "L'économie",
    "Le gouvernement"
]

print("\nGenerating sample texts...")
sample_generations = {}
for seed in sample_seeds:
    generated_text = generate_text(lstm_model, vocab, tokenizer, seed, max_length=30)
    sample_generations[seed] = generated_text
    print(f"\nSeed: {seed}")
    print(f"Generated: {generated_text}")

try:
    # Save plots first
    save_training_plots(exp_dir, train_losses, test_losses, perplexities, sample_generations)
    print("Plots saved successfully")
except Exception as e:
    print(f"Warning: Could not save plots: {e}")

try:
    # Save training results
    results = {
        "hyperparameters": hyperparams,
        "training_stats": {
            "total_training_time": f"{training_time:.2f} seconds",
            "final_train_loss": float(train_losses[-1]),
            "final_test_loss": float(test_losses[-1]),
            "final_perplexity": float(perplexities[-1]),
            "best_perplexity": float(min(perplexities)),
            "total_parameters": total_params,
            "train_losses": [float(l) for l in train_losses],
            "test_losses": [float(l) for l in test_losses],
            "perplexities": [float(p) for p in perplexities]
        },
        "sample_generations": sample_generations
    }

    # Save results to JSON
    with open(os.path.join(exp_dir, "training_results.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Results saved successfully")
except Exception as e:
    print(f"Warning: Could not save results: {e}")

try:
    # Save final model
    model_save_path = os.path.join(exp_dir, "final_model.pth")
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'vocab_size': len(vocab),
        'hyperparameters': hyperparams,
        'sample_generations': sample_generations  # Also save the generations with the model
    }, model_save_path)
    print(f"Model saved successfully to {model_save_path}")
except Exception as e:
    print(f"Warning: Could not save model: {e}")

print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"All results saved in: {exp_dir}")