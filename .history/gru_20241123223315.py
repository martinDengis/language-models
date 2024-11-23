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
from lstm import create_experiment_dir, save_training_plots, calculate_perplexity, generate_text, hyperparams
warnings.filterwarnings('ignore')

# Set up the environment and get the device
device = setup_environment()
# Create experiment directory
exp_dir = create_experiment_dir(prefix="gru_h{hidden_size}_l{num_layers}")

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


# ### GRU Init

# In[ ]:


# Initialize GRU Model
print("Initializing GRU model...")
gru_model = GRU(num_emb=len(vocab),
                output_size=len(vocab),
                num_layers=num_layers,
                hidden_size=hidden_size).to(device)

# Print model summary
print("\nGRU Model Architecture:")
print(gru_model)

# Calculate total parameters
gru_params = sum(p.numel() for p in gru_model.parameters())
print(f"\nTotal GRU parameters: {gru_params:,}")

# Initialize optimizer and loss function for GRU
gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)
gru_loss_fn = nn.CrossEntropyLoss()


# ### GRU Training

# In[ ]:


# Training and Validation Setup for GRU
# Initialize lists to store metrics
gru_train_losses = []
gru_test_losses = []
gru_perplexities = []

# GRU Training Loop
print("\nStarting GRU Training...")
gru_start_time = time.time()
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


# ## Visualise the results (LSTM vsGRU)

# In[ ]:


# Comparison Visualization
plt.figure(figsize=(15, 10))

# Plot training losses
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='LSTM Train Loss')
plt.plot(gru_train_losses, label='GRU Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# Plot validation losses
plt.subplot(2, 2, 2)
plt.plot(test_losses, label='LSTM Test Loss')
plt.plot(gru_test_losses, label='GRU Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()

# Plot perplexities
plt.subplot(2, 2, 3)
plt.plot(perplexities, label='LSTM Perplexity')
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
    f'Training Loss: {train_losses[-1]:.4f}\n'
    f'Validation Loss: {test_losses[-1]:.4f}\n'
    f'Perplexity: {perplexities[-1]:.2f}\n'
    f'Training Time: {training_time:.2f}s\n'
    f'Parameters: {total_params:,}\n\n'
    f'GRU Final Metrics:\n'
    f'Training Loss: {gru_train_losses[-1]:.4f}\n'
    f'Validation Loss: {gru_test_losses[-1]:.4f}\n'
    f'Perplexity: {gru_perplexities[-1]:.2f}\n'
    f'Training Time: {gru_training_time:.2f}s\n'
    f'Parameters: {gru_params:,}'
)
plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()


# In[ ]:


# Print final comparison
print("\nFinal Comparison Summary:")
print("\nLSTM Model:")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {test_losses[-1]:.4f}")
print(f"Final Perplexity: {perplexities[-1]:.2f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Number of Parameters: {total_params:,}")

print("\nGRU Model:")
print(f"Final Training Loss: {gru_train_losses[-1]:.4f}")
print(f"Final Validation Loss: {gru_test_losses[-1]:.4f}")
print(f"Final Perplexity: {gru_perplexities[-1]:.2f}")
print(f"Training Time: {gru_training_time:.2f} seconds")
print(f"Number of Parameters: {gru_params:,}")

