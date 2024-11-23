import matplotlib.pyplot as plt
import os
import json

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



def compare_models(lstm_results, gru_results, lstm_params, gru_params):
    lstm_train_losses, lstm_test_losses, lstm_perplexities, lstm_training_time = lstm_results
    gru_train_losses, gru_test_losses, gru_perplexities, gru_training_time = gru_results

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
        f'Parameters: {lstm_params:,}\n\n'
        f'GRU Final Metrics:\n'
        f'Training Loss: {gru_train_losses[-1]:.4f}\n'
        f'Validation Loss: {gru_test_losses[-1]:.4f}\n'
        f'Perplexity: {gru_perplexities[-1]:.2f}\n'
        f'Training Time: {gru_training_time:.2f}s\n'
        f'Parameters: {gru_params:,}'
    )
    plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
    plt.show()