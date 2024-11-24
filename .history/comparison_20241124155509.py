# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from config import learning_rate, nepochs, hidden_size, num_layers, batch_size, max_len
import os
import json
import glob
import warnings

import re
warnings.filterwarnings('ignore')

print("Running comparison.py...")

# Define base directory for LSTM results
lstm_results_dir = "experiments"

# Ensure the directory exists
if not os.path.exists(lstm_results_dir):
    raise FileNotFoundError(f"The directory '{lstm_results_dir}' does not exist.")

# Get all training_results.json paths recursively
lstm_results_files = glob.glob(os.path.join(lstm_results_dir, "**", "training_results.json"), recursive=True)

if not lstm_results_files:
    raise FileNotFoundError("No LSTM results files found in the specified directory.")

# Debug: List all files found
print(f"Found LSTM results files: {lstm_results_files}")

# Filter LSTM results by matching hidden_size and num_layers
matching_results = []
for path in lstm_results_files:
    # Extract directory name (where hidden_size and num_layers are encoded)
    lstm_dir_name = os.path.basename(os.path.dirname(path))
    match = re.search(r"lstm_h(\d+)_l(\d+)", lstm_dir_name)
    if match:
        lstm_hidden_size = int(match.group(1))
        lstm_num_layers = int(match.group(2))
        
        # Check if LSTM hyperparameters match GRU's
        if lstm_hidden_size == hidden_size and lstm_num_layers == num_layers:
            # Append the path along with the timestamp extracted from the directory name
            timestamp_match = re.search(r"\d{8}_\d{6}", lstm_dir_name)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(), "%Y%m%d_%H%M%S")
                matching_results.append((path, timestamp))

# Sort matching results by timestamp (most recent first)
matching_results.sort(key=lambda x: x[1], reverse=True)

if not matching_results:
    raise FileNotFoundError(
        f"No LSTM results found matching hidden_size={hidden_size} and num_layers={num_layers}."
)

# Use the most recent matching LSTM results
latest_lstm_results_path, latest_timestamp = matching_results[0]

# Debug: Confirm which LSTM result is being used
print(f"Using LSTM results from: {latest_lstm_results_path}")
print(f"Timestamp: {latest_timestamp}")

# Load the results JSON
with open(latest_lstm_results_path, "r", encoding="utf-8") as f:
    lstm_results = json.load(f)

# Extract relevant training stats from the LSTM results
lstm_train_losses = lstm_results["training_stats"]["train_losses"]
lstm_test_losses = lstm_results["training_stats"]["test_losses"]
lstm_perplexities = lstm_results["training_stats"]["perplexities"]
lstm_training_time = lstm_results["training_stats"]["total_training_time"]
lstm_total_params = lstm_results["training_stats"]["total_parameters"]

# Debug: Print out the loaded stats
print(f"LSTM Training Losses: {lstm_train_losses}")
print(f"LSTM Test Losses: {lstm_test_losses}")
print(f"LSTM Perplexities: {lstm_perplexities}")
print(f"LSTM Training Time: {lstm_training_time}")
print(f"LSTM Total Parameters: {lstm_total_params}")



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

lstm_training_time = float(lstm_training_time.split()[0])  # Convert "88.98 seconds" -> 88.98

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

