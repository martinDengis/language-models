import os
import json
import matplotlib.pyplot as plt

def save_and_plot_results(model_name, train_losses, test_losses, perplexities, training_time):
    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "perplexities": perplexities,
        "training_time": training_time
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_results.json", "w") as f:
        json.dump(results, f)

    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    # Plot perplexity
    plt.subplot(1, 2, 2)
    plt.plot(perplexities, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Model Perplexity')
    plt.tight_layout()

    # Save the plots
    plt.savefig(f"results/{model_name}_training_summary.png")
    plt.show()