import json
import matplotlib.pyplot as plt
import argparse
import os

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_comparison(lstm_data, gru_data, save_dir):
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(lstm_data['training_stats']['train_losses'], label='LSTM Train Loss')
    plt.plot(gru_data['training_stats']['train_losses'], label='GRU Train Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Test Loss
    plt.subplot(2, 2, 2)
    plt.plot(lstm_data['training_stats']['test_losses'], label='LSTM Test Loss')
    plt.plot(gru_data['training_stats']['test_losses'], label='GRU Test Loss')
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 3: Perplexity
    plt.subplot(2, 2, 3)
    plt.plot(lstm_data['training_stats']['perplexities'], label='LSTM Perplexity')
    plt.plot(gru_data['training_stats']['perplexities'], label='GRU Perplexity')
    plt.title('Perplexity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)

    # Text summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""Model Comparison Summary:

LSTM:
Hyperparameters:
- Learning rate: {lstm_data['hyperparameters']['learning_rate']}
- Epochs: {lstm_data['hyperparameters']['nepochs']}
- Batch size: {lstm_data['hyperparameters']['batch_size']}
- Hidden size: {lstm_data['hyperparameters']['hidden_size']}
- Num layers: {lstm_data['hyperparameters']['num_layers']}
Total parameters: {lstm_data['training_stats']['total_parameters']:,}
Best perplexity: {lstm_data['training_stats']['best_perplexity']:.2f}
Training time: {float(lstm_data['training_stats']['total_training_time'].split()[0]):.2f} seconds

GRU:
Hyperparameters:
- Learning rate: {gru_data['hyperparameters']['learning_rate']}
- Epochs: {gru_data['hyperparameters']['nepochs']}
- Batch size: {gru_data['hyperparameters']['batch_size']}
- Hidden size: {gru_data['hyperparameters']['hidden_size']}
- Num layers: {gru_data['hyperparameters']['num_layers']}
Total parameters: {gru_data['training_stats']['total_parameters']:,}
Best perplexity: {gru_data['training_stats']['best_perplexity']:.2f}
Training time: {float(gru_data['training_stats']['total_training_time'].split()[0]):.2f} seconds"""

    plt.text(0, 1, summary_text, fontsize=8, verticalalignment='top')

    # Adjust layout and save
    plt.tight_layout()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare training results between LSTM and GRU models')
    parser.add_argument('lstm_path', type=str, help='Path to LSTM training results JSON')
    parser.add_argument('gru_path', type=str, help='Path to GRU training results JSON')
    parser.add_argument('--save_dir', type=str, default='plots', help='Directory to save the plots')
    
    args = parser.parse_args()
    
    # Load data
    lstm_data = load_json(args.lstm_path)
    gru_data = load_json(args.gru_path)
    
    # Generate plots
    plot_comparison(lstm_data, gru_data, args.save_dir)

if __name__ == "__main__":
    main()