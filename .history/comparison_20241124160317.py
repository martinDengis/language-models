# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import json
import glob
import warnings
from datetime import datetime
import re
from config import learning_rate, nepochs, hidden_size, num_layers, batch_size, max_len


warnings.filterwarnings('ignore')

print("Running comparison.py...")

# Define base directory for LSTM results
results_dir = "experiments"

# Ensure the directory exists
if not os.path.exists(results_dir):
    raise FileNotFoundError(f"The directory '{results_dir}' does not exist.")

# Helper function to load the most recent results for a model
def load_latest_results(model_name):
    model_results_files = glob.glob(os.path.join(results_dir, "**", "training_results.json"), recursive=True)

    # Filter files by model name in the directory path
    matching_results = []
    for path in model_results_files:
        dir_name = os.path.basename(os.path.dirname(path))
        if model_name in dir_name:
            # Extract timestamp from directory name for sorting
            timestamp_match = re.search(r"\d{8}_\d{6}", dir_name)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(), "%Y%m%d_%H%M%S")
                matching_results.append((path, timestamp))

    # Sort matching results by timestamp (most recent first)
    matching_results.sort(key=lambda x: x[1], reverse=True)

    if not matching_results:
        raise FileNotFoundError(f"No results found for model '{model_name}' in '{results_dir}'.")

    # Use the most recent matching results
    return matching_results[0][0]  # Return the path to the latest result file

# Get all training_results.json paths recursively
lstm_results_files = glob.glob(os.path.join(results_dir, "**", "training_results.json"), recursive=True)

if not lstm_results_files:
    raise FileNotFoundError("No LSTM results files found in the specified directory.")

# Load results for both LSTM and GRU
print("Loading LSTM results...")
lstm_results_path = load_latest_results("lstm")
with open(lstm_results_path, "r", encoding="utf-8") as f:
    lstm_results = json.load(f)

print("Loading GRU results...")
gru_results_path = load_latest_results("gru")
with open(gru_results_path, "r", encoding="utf-8") as f:
    gru_results = json.load(f)

# Extract metrics
lstm_train_losses = lstm_results["training_stats"]["train_losses"]
lstm_test_losses = lstm_results["training_stats"]["test_losses"]
lstm_perplexities = lstm_results["training_stats"]["perplexities"]
lstm_training_time = float(lstm_results["training_stats"]["total_training_time"].split()[0])  # Convert "88.98 seconds" -> 88.98
lstm_total_params = lstm_results["training_stats"]["total_parameters"]

gru_train_losses = gru_results["training_stats"]["train_losses"]
gru_test_losses = gru_results["training_stats"]["test_losses"]
gru_perplexities = gru_results["training_stats"]["perplexities"]
gru_training_time = float(gru_results["training_stats"]["total_training_time"].split()[0])  # Convert "88.98 seconds" -> 88.98
gru_total_params = gru_results["training_stats"]["total_parameters"]
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
gru_training_time = float(gru_training_time.split()[0])  # Convert "88.98 seconds" -> 88.98

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
comparison_plot_path = os.path.join(results_dir, "comparison_plot.pdf")
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

comparison_values_path = os.path.join(results_dir, "comparison_values.json")
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

