# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import time
import os
import json
from datetime import datetime
import spacy
import warnings
from setup import setup_environment
from config import learning_rate, nepochs, batch_size, max_len, hidden_size, num_layers
from gensim.models import Word2Vec
warnings.filterwarnings('ignore')

print("Running exp2.py...")

# Set up the environment and get the device
device = setup_environment()

# Generate multiple sample texts with different seeds
sample_seeds = [
    "Le président",
    "La France",
    "Les électeurs",
    "L'économie",
    "Le gouvernement"
]

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

def create_experiment_dir(base_dir="experiments", sub_folder="exp2", prefix="run"):
    """
    Create a dedicated folder for the current experiment.
    The structure will look like:
    experiments/exp2/run_YYYYMMDD_HHMMSS/
    """
    # Ensure the base directory and sub-folder exist
    dedicated_dir = os.path.join(base_dir, sub_folder)
    if not os.path.exists(dedicated_dir):
        os.makedirs(dedicated_dir)

    # Add timestamp for unique directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(dedicated_dir, f"{prefix}_{timestamp}")
    os.makedirs(exp_dir)

    return exp_dir

def save_training_plots(exp_dir, train_losses, test_losses, perplexities, sample_generations=None):
    """Save training plots, sample generations, and metrics."""
    try:
        # Convert lists to numpy arrays explicitly
        train_losses = np.array([float(x) for x in train_losses])
        test_losses = np.array([float(x) for x in test_losses])
        perplexities = np.array([float(x) for x in perplexities])
        
        # Create figure
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
        
        # Save the plots
        plot_png_path = os.path.join(exp_dir, "training_plots.png")
        plt.savefig(plot_png_path, format='png', bbox_inches='tight')
        plt.close()

        # Save raw data as CSV
        import pandas as pd
        metrics_df = pd.DataFrame({
            'epoch': np.arange(len(train_losses)),
            'train_loss': train_losses,
            'test_loss': test_losses,
            'perplexity': perplexities
        })
        metrics_csv_path = os.path.join(exp_dir, "training_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Training metrics saved to {metrics_csv_path}")
        
        # Save sample generations (if provided)
        if sample_generations:
            sample_path = os.path.join(exp_dir, "sample_generations.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                for seed, text in sample_generations.items():
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Generated: {text}\n")
                    f.write("-" * 50 + "\n")
            print(f"Sample generations saved to {sample_path}")
    
    except Exception as e:
        print(f"Warning: Could not save plots due to error: {str(e)}")
        print("Saving raw data instead...")
        # Save raw data as a fallback
        with open(os.path.join(exp_dir, "training_metrics.txt"), "w") as f:
            f.write(f"Train Losses: {train_losses.tolist()}\n")
            f.write(f"Test Losses: {test_losses.tolist()}\n")
            f.write(f"Perplexities: {perplexities.tolist()}\n")

# Create experiment directory
exp_dir = create_experiment_dir(base_dir="experiments", sub_folder="exp2", prefix="run")
print(f"All results will be saved in: {exp_dir}")

# Experiment 2: Investigate the performance using 1-hot encoding and Word2Vec as input.

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

# Word2Vec Encoder
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

# Training Functions

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
    
    # Add timing
    start_time = time.time()

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
        
    total_training_time = time.time() - start_time

    return train_losses, test_losses, perplexities, total_training_time

# Run Experiments

# Initialize and train all models
def run_experiments(train_loader, test_loader, vocab, device, config, exp_dir):
    """
    Run experiments with different models and embeddings.
    Save model metrics and checkpoints to the experiment directory.
    """
    results = {}

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
        print(f"\nTraining {name}")

        # Initialize optimizer and loss function
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()

        # Train the model
        train_losses, test_losses, perplexities, training_time = train_model(
            model, train_loader, test_loader, optimizer, loss_fn,
            config['num_epochs'], device, name
        )

        # Save metrics
        model_results_path = os.path.join(exp_dir, f"results_{name}.json")
        results_dict = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "perplexities": perplexities,
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1],
            "final_perplexity": perplexities[-1],
            "total_training_time": f"{training_time:.2f} seconds",
            "hyperparameters": {
                "learning_rate": config['learning_rate'],
                "nepochs": config['num_epochs'],
                "batch_size": batch_size,
                "max_len": max_len,
                "hidden_size": config['hidden_size'],
                "num_layers": config['num_layers']
            },
            "total_parameters": sum(p.numel() for p in model.parameters())
        }
        
        with open(model_results_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results for {name} saved to {model_results_path}")

        # Save model checkpoint
        model_checkpoint_path = os.path.join(exp_dir, f"model_{name}.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses,
            "perplexities": perplexities,
            "training_time": training_time
        }, model_checkpoint_path)
        print(f"Model checkpoint for {name} saved to {model_checkpoint_path}")

        # Store metrics for final results comparison
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'perplexities': perplexities,
            'training_time': training_time
        }

    return results

# Visualization of results
def plot_comparison(results, exp_dir):
    """Plot comparison of all models and save to the experiment directory."""
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

    # Updated summary text to include training time
    summary_text = "Final Metrics Summary\n\n"
    for name, metrics in results.items():
        summary_text += f"{name}:\n"
        summary_text += f"Final Train Loss: {metrics['train_losses'][-1]:.4f}\n"
        summary_text += f"Final Valid Loss: {metrics['test_losses'][-1]:.4f}\n"
        summary_text += f"Final Perplexity: {metrics['perplexities'][-1]:.2f}\n"
        summary_text += f"Training Time: {metrics['training_time']:.2f} seconds\n\n"
    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top')

    plt.tight_layout()

    # Save plots
    comparison_plot_path = os.path.join(exp_dir, "comparison_plot.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Comparison plot saved to {comparison_plot_path}")


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

results = run_experiments(train_loader, test_loader, vocab, device, config, exp_dir)

# Step 4: Plot and save results
plot_comparison(results, exp_dir)

# Print final comparison
print("\nFinal Results Summary:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"Final Training Loss: {metrics['train_losses'][-1]:.4f}")
    print(f"Final Validation Loss: {metrics['test_losses'][-1]:.4f}")
    print(f"Final Perplexity: {metrics['perplexities'][-1]:.2f}")

