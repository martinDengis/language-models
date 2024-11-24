# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
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
from config import learning_rate, nepochs, hidden_size, num_layers, batch_size, max_len
from lstm import calculate_perplexity, generate_text, save_training_plots
import re
import pickle
warnings.filterwarnings('ignore')

print("Running gru.py...")

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

# Store hyperparameters
hyperparams = {
    "learning_rate": learning_rate,
    "nepochs": nepochs,
    "batch_size": batch_size,
    "max_len": max_len,
    "hidden_size": hidden_size,
    "num_layers": num_layers
}

# Generate multiple sample texts with different seeds
sample_seeds = [
    "Le président",
    "La France",
    "Les électeurs",
    "L'économie",
    "Le gouvernement"
]

# Load the latest LSTM results
lstm_results_dir = "experiments"
if not os.path.exists(lstm_results_dir):
    raise FileNotFoundError(f"The directory '{lstm_results_dir}' does not exist.")
lstm_results_files = sorted([f for f in os.listdir(lstm_results_dir) if f.startswith("lstm") and f.endswith("training_results.json")], reverse=True)

if not lstm_results_files:
    raise FileNotFoundError("No LSTM results files found in the specified directory.")

latest_lstm_results_path = os.path.join(lstm_results_dir, lstm_results_files[0])

# Extract hyperparameters from the LSTM filename
match = re.search(r"lstm_h(\d+)_l(\d+)", latest_lstm_results_path)
if match:
    hidden_size = int(match.group(1))
    num_layers = int(match.group(2))
else:
    raise ValueError("Could not extract hyperparameters from LSTM filename")

# Create experiment directory for GRU
exp_dir = create_experiment_dir(prefix=f"gru_h{hidden_size}_l{num_layers}")

# Load the LSTM results
with open(latest_lstm_results_path, "r", encoding='utf-8') as f:
    lstm_results = json.load(f)

# Extract the necessary values from the LSTM results
lstm_train_losses = lstm_results["training_stats"]["train_losses"]
lstm_test_losses = lstm_results["training_stats"]["test_losses"]
lstm_perplexities = lstm_results["training_stats"]["perplexities"]
lstm_training_time = lstm_results["training_stats"]["total_training_time"]
lstm_total_params = lstm_results["training_stats"]["total_parameters"]

print(f"Loaded LSTM results from {latest_lstm_results_path}")
print(f"Hyperparameters: hidden_size={hidden_size}, num_layers={num_layers}")
print(f"LSTM Training Losses: {lstm_train_losses}")
print(f"LSTM Test Losses: {lstm_test_losses}")
print(f"LSTM Perplexities: {lstm_perplexities}")
print(f"LSTM Training Time: {lstm_training_time}")
print(f"LSTM Total Parameters: {lstm_total_params}")

# Load the vocabulary and tokenizer
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

def generate_text(model, vocab, tokenizer, seed_text, max_length=20):
    model.eval()
    device = next(model.parameters()).device

    # Tokenize seed text
    tokens = tokenizer(seed_text)
    current_sequence = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

    # Initialize hidden states
    hidden = torch.zeros(num_layers, 1, hidden_size, device=device)

    # Initialize memory cell only for LSTM
    #memory = torch.zeros(num_layers, 1, hidden_size, device=device) if isinstance(model, LSTM) else None

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_sequence = current_sequence[-1:].unsqueeze(0).to(device)

            # Get model prediction based on model type
            #if isinstance(model, LSTM):
                #output, hidden, memory = model(input_sequence, hidden, memory)
            #else:  # GRU case
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

# Initialize GRU Model
print("Initializing GRU model...")
gru_model = GRU(num_emb=len(vocab),
                output_size=len(vocab),
                num_layers=num_layers,
                hidden_size=hidden_size).to(device)

# Initialize optimizer and loss function for GRU
gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

# Print model summary
print("\nGRU Model Architecture:")
print(gru_model)

# Calculate total parameters
gru_total_params = sum(p.numel() for p in gru_model.parameters())
print(f"\nTotal GRU parameters: {gru_total_params:,}")

# Loss function
gru_loss_fn = nn.CrossEntropyLoss()

# Training Loop
gru_start_time = time.time()
gru_train_losses = []
gru_test_losses = []
gru_perplexities = []

# GRU Training Loop
print("\nStarting GRU Training...")
# Initialize the progress bar
gru_epoch_bar = trange(nepochs, desc="GRU Training Progress")

for epoch in gru_epoch_bar:
    # Training phase
    gru_model.train()
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

        # Forward pass
        output, hidden = gru_model(sequences, hidden)
        output = output.view(-1, len(vocab))
        targets = targets.view(-1)

        # Calculate loss
        loss = gru_loss_fn(output, targets)
        train_loss += loss.item()
        train_steps += 1

        # Backward pass
        gru_optimizer.zero_grad()
        loss.backward()
        gru_optimizer.step()

        # Update progress bar
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / train_steps
    gru_train_losses.append(avg_train_loss)

    # Validation Phase
    gru_model.eval()
    test_loss = 0
    test_steps = 0

    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc="Validation", leave=False):
            sequences = sequences.to(device)
            targets = targets.to(device)
            bs = sequences.shape[0]
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)

            output, hidden = gru_model(sequences, hidden)
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)

            loss = gru_loss_fn(output, targets)
            test_loss += loss.item()
            test_steps += 1

    avg_test_loss = test_loss / test_steps
    gru_test_losses.append(avg_test_loss)

    # Calculate perplexity
    perplexity = calculate_perplexity(avg_test_loss)
    gru_perplexities.append(perplexity)

    # Update progress bar
    gru_epoch_bar.set_postfix(
        train_loss=f"{avg_train_loss:.4f}",
        test_loss=f"{avg_test_loss:.4f}",
        perplexity=f"{perplexity:.2f}"
    )

    # Generate sample text every few epochs
    if (epoch + 1) % 2 == 0:
        sample_text = generate_text(gru_model, vocab, tokenizer, "Le président", max_length=20)
        print(f"\nGRU Sample text generation: {sample_text}\n")

gru_training_time = time.time() - gru_start_time

print("\nGenerating sample texts...")
sample_generations = {}
for seed in sample_seeds:
    generated_text = generate_text(gru_model, vocab, tokenizer, seed, max_length=30)
    sample_generations[seed] = generated_text
    print(f"\nSeed: {seed}")
    print(f"Generated: {generated_text}")

try:
    # Save plots first
    save_training_plots(exp_dir, gru_train_losses, gru_test_losses, gru_perplexities, sample_generations)
    print("Plots saved successfully")
except Exception as e:
    print(f"Warning: Could not save plots: {e}")

try:
    # Save training results
    gru_results = {
        "hyperparameters": hyperparams,
        "training_stats": {
            "total_training_time": f"{gru_training_time:.2f} seconds",
            "final_train_loss": float(gru_train_losses[-1]),
            "final_test_loss": float(gru_test_losses[-1]),
            "final_perplexity": float(gru_perplexities[-1]),
            "best_perplexity": float(min(gru_perplexities)),
            "total_parameters": gru_total_params,
            "train_losses": [float(l) for l in gru_train_losses],
            "test_losses": [float(l) for l in gru_test_losses],
            "perplexities": [float(p) for p in gru_perplexities]
        },
        "sample_generations": sample_generations
    }

    # Save results to JSON
    with open(os.path.join(exp_dir, "training_results.json"), "w", encoding='utf-8') as f:
        json.dump(gru_results, f, indent=4, ensure_ascii=False)
    print("Results saved successfully")
except Exception as e:
    print(f"Warning: Could not save results: {e}")

try:
    # Save final model
    model_save_path = os.path.join(exp_dir, "final_model.pth")
    torch.save({
        'model_state_dict': gru_model.state_dict(),
        'vocab_size': len(vocab),
        'hyperparameters': hyperparams,
        'sample_generations': sample_generations  # Also save the generations with the model
    }, model_save_path)
    print(f"Model saved successfully to {model_save_path}")
except Exception as e:
    print(f"Warning: Could not save model: {e}")

print(f"\nTraining completed in {gru_training_time:.2f} seconds")
print(f"All results saved in: {exp_dir}")


# Visualise the results (LSTM vsGRU)

# Comparison Visualization
plt.figure(figsize=(15, 10))

# Plot training losses
plt.subplot(2, 2, 1)
plt.plot(lstm_train_losses, label='LSTM Train Loss')
plt.plot(gru_train_losses, label='GRU Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# Plot validation losses
plt.subplot(2, 2, 2)
plt.plot(lstm_test_losses, label='LSTM Test Loss')
plt.plot(gru_test_losses, label='GRU Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()

# Plot perplexities
plt.subplot(2, 2, 3)
plt.plot(lstm_perplexities, label='LSTM Perplexity')
plt.plot(gru_perplexities, label='GRU Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison')
plt.legend()

# Add comparison metrics
plt.subplot(2, 2, 4)
plt.axis('off')
info_text = (
    f'Model Comparison Summary\n\n'
    f'LSTM Final Metrics:\n'
    f'Training Loss: {lstm_train_losses[-1]:.4f}\n'
    f'Validation Loss: {lstm_test_losses[-1]:.4f}\n'
    f'Perplexity: {lstm_perplexities[-1]:.2f}\n'
    f'Training Time: {lstm_training_time:.2f}s\n'
    f'Parameters: {lstm_total_params:,}\n\n'
    f'GRU Final Metrics:\n'
    f'Training Loss: {gru_train_losses[-1]:.4f}\n'
    f'Validation Loss: {gru_test_losses[-1]:.4f}\n'
    f'Perplexity: {gru_perplexities[-1]:.2f}\n'
    f'Training Time: {gru_training_time:.2f}s\n'
    f'Parameters: {gru_total_params:,}'
)
plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')

plt.tight_layout()
# Save the plot as a PDF
comparison_plot_path = os.path.join(exp_dir, "comparison_plot.pdf")
plt.savefig(comparison_plot_path)
print(f"Comparison plot saved to {comparison_plot_path}")

plt.show()

# Save final comparison values to a JSON file
comparison_values = {
    "LSTM": {
        "Final Training Loss": lstm_train_losses[-1],
        "Final Validation Loss": lstm_test_losses[-1],
        "Final Perplexity": lstm_perplexities[-1],
        "Training Time": lstm_training_time,
        "Number of Parameters": lstm_total_params
    },
    "GRU": {
        "Final Training Loss": gru_train_losses[-1],
        "Final Validation Loss": gru_test_losses[-1],
        "Final Perplexity": gru_perplexities[-1],
        "Training Time": gru_training_time,
        "Number of Parameters": gru_total_params
    }
}

comparison_values_path = os.path.join(exp_dir, "comparison_values.json")
with open(comparison_values_path, "w", encoding='utf-8') as f:
    json.dump(comparison_values, f, indent=4, ensure_ascii=False)
print(f"Comparison values saved to {comparison_values_path}")

# Print final comparison
print("\nFinal Comparison Summary:")
print("\nLSTM Model:")
print(f"Final Training Loss: {lstm_train_losses[-1]:.4f}")
print(f"Final Validation Loss: {lstm_test_losses[-1]:.4f}")
print(f"Final Perplexity: {lstm_perplexities[-1]:.2f}")
print(f"Training Time: {lstm_training_time:.2f} seconds")
print(f"Number of Parameters: {lstm_total_params:,}")

print("\nGRU Model:")
print(f"Final Training Loss: {gru_train_losses[-1]:.4f}")
print(f"Final Validation Loss: {gru_test_losses[-1]:.4f}")
print(f"Final Perplexity: {gru_perplexities[-1]:.2f}")
print(f"Training Time: {gru_training_time:.2f} seconds")
print(f"Number of Parameters: {gru_total_params:,}")

