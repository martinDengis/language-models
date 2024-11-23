import matplotlib.pyplot as plt
import os
import json
import torch

def save_and_plot_results(model_name, train_losses, test_losses, perplexities, training_time):
    """
    Saves training results to a file and plots them.

    Args:
        model_name (str): Name of the model (used for saving files).
        train_losses (list): List of training losses over epochs.
        test_losses (list): List of test losses over epochs.
        perplexities (list): List of perplexities over epochs.
        training_time (float): Total training time in seconds.
    """
    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "perplexities": perplexities,
        "training_time": training_time
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(results, f)

    # Plot losses and perplexity
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Perplexity plot
    plt.subplot(1, 2, 2)
    plt.plot(perplexities, label="Perplexity", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Perplexity")
    plt.legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_training_summary.png")
    plt.show()

def calculate_perplexity(loss):
    """
    Calculates perplexity from a given loss value.

    Args:
        loss (float): The loss value (typically the average test loss).

    Returns:
        float: Perplexity value.
    """
    return torch.exp(torch.tensor(loss)).item()

def compare_model_results(lstm_results, gru_results, lstm_params, gru_params, lstm_time, gru_time):
    """
    Visualizes and compares the training results of LSTM and GRU models.

    Args:
        lstm_results (tuple): LSTM results containing (train_losses, test_losses, perplexities).
        gru_results (tuple): GRU results containing (train_losses, test_losses, perplexities).
        lstm_params (int): Number of parameters in the LSTM model.
        gru_params (int): Number of parameters in the GRU model.
        lstm_time (float): Total training time for the LSTM model.
        gru_time (float): Total training time for the GRU model.
    """
    lstm_train_losses, lstm_test_losses, lstm_perplexities = lstm_results
    gru_train_losses, gru_test_losses, gru_perplexities = gru_results

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
        f'Training Time: {lstm_time:.2f}s\n'
        f'Parameters: {lstm_params:,}\n\n'
        f'GRU Final Metrics:\n'
        f'Training Loss: {gru_train_losses[-1]:.4f}\n'
        f'Validation Loss: {gru_test_losses[-1]:.4f}\n'
        f'Perplexity: {gru_perplexities[-1]:.2f}\n'
        f'Training Time: {gru_time:.2f}s\n'
        f'Parameters: {gru_params:,}'
    )
    plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def plot_training_summary(train_losses, test_losses, perplexities, training_time, model_name):
    """
    Plots the training summary for a model.

    Args:
        train_losses (list): Training losses over epochs.
        test_losses (list): Validation losses over epochs.
        perplexities (list): Perplexity values over epochs.
        training_time (float): Total training time in seconds.
        model_name (str): Name of the model being trained.
    """
    print(f"\nTraining completed in {training_time:.2f} seconds for {model_name}")

    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()

    # Plot perplexity
    plt.subplot(1, 2, 2)
    plt.plot(perplexities, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f'{model_name} Model Perplexity')
    plt.tight_layout()

    plt.show()
