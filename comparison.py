# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import json
import glob
import warnings
from datetime import datetime
import re
from config import hidden_size, num_layers # Import hyperparameters from config.py
warnings.filterwarnings('ignore')

print("Running comparison.py...")

# Define base directory for results
results_dir = "experiments"
comparison_dir = os.path.join(results_dir, "comparison")

# Ensure the base directory exists
if not os.path.exists(results_dir):
    raise FileNotFoundError(f"The directory '{results_dir}' does not exist.")

# Ensure the comparison directory exists
os.makedirs(comparison_dir, exist_ok=True)

# Helper function to load the most recent matching results for a model
def load_matching_results(model_prefix, hidden_size, num_layers):
    model_results_files = glob.glob(os.path.join(results_dir, "**", "training_results.json"), recursive=True)

    # Filter files by model name and matching hyperparameters
    matching_results = []
    for path in model_results_files:
        dir_name = os.path.basename(os.path.dirname(path))
        if model_prefix in dir_name:
            # Check for matching hidden_size and num_layers
            match = re.search(rf"{model_prefix}_h(\d+)_l(\d+)", dir_name)
            if match:
                result_hidden_size = int(match.group(1))
                result_num_layers = int(match.group(2))
                if result_hidden_size == hidden_size and result_num_layers == num_layers:
                    # Extract timestamp from directory name for sorting
                    timestamp_match = re.search(r"\d{8}_\d{6}", dir_name)
                    if timestamp_match:
                        timestamp = datetime.strptime(timestamp_match.group(), "%Y%m%d_%H%M%S")
                        matching_results.append((path, timestamp))

    # Sort matching results by timestamp (most recent first)
    matching_results.sort(key=lambda x: x[1], reverse=True)

    if not matching_results:
        raise FileNotFoundError(
            f"No results found for '{model_prefix}' with hidden_size={hidden_size} and num_layers={num_layers}."
        )

    # Use the most recent matching results
    return matching_results[0][0]  # Return the path to the latest result file

# Load LSTM and GRU results
print(f"Loading LSTM results with hidden_size={hidden_size} and num_layers={num_layers}...")
lstm_results_path = load_matching_results("lstm", hidden_size, num_layers)
with open(lstm_results_path, "r", encoding="utf-8") as f:
    lstm_results = json.load(f)

print(f"Loading GRU results with hidden_size={hidden_size} and num_layers={num_layers}...")
gru_results_path = load_matching_results("gru", hidden_size, num_layers)
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

# Visualize the results (LSTM vs GRU)

# Comparison Visualization
plt.figure(figsize=(15, 10))

# Plot training losses
plt.subplot(2, 2, 1)
plt.plot(lstm_train_losses, label='LSTM Train Loss')
plt.plot(gru_train_losses, label='LSTM Attention Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# Plot validation losses
plt.subplot(2, 2, 2)
plt.plot(lstm_test_losses, label='LSTM Test Loss')
plt.plot(gru_test_losses, label='LSTM Attention Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()

# Plot perplexities
plt.subplot(2, 2, 3)
plt.plot(lstm_perplexities, label='LSTM Perplexity')
plt.plot(gru_perplexities, label='LSTM Attention Perplexity')
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
    f'LSTM Attention Final Metrics:\n'
    f'Training Loss: {gru_train_losses[-1]:.4f}\n'
    f'Validation Loss: {gru_test_losses[-1]:.4f}\n'
    f'Perplexity: {gru_perplexities[-1]:.2f}\n'
    f'Training Time: {gru_training_time:.2f}s\n'
    f'Parameters: {gru_total_params:,}'
)
plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')

plt.tight_layout()

# Save the plot as a PDF
comparison_plot_path = os.path.join(comparison_dir, "comparison_plot.pdf")
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
    "LSTM Attention": {
        "Final Training Loss": gru_train_losses[-1],
        "Final Validation Loss": gru_test_losses[-1],
        "Final Perplexity": gru_perplexities[-1],
        "Training Time": gru_training_time,
        "Number of Parameters": gru_total_params
    }
}

comparison_values_path = os.path.join(comparison_dir, "comparison_values.json")
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

print("\nLSTM Attention Model:")
print(f"Final Training Loss: {gru_train_losses[-1]:.4f}")
print(f"Final Validation Loss: {gru_test_losses[-1]:.4f}")
print(f"Final Perplexity: {gru_perplexities[-1]:.2f}")
print(f"Training Time: {gru_training_time:.2f} seconds")
print(f"Number of Parameters: {gru_total_params:,}")
