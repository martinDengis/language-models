from setup import setup_environment
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from experiments import experiment_1, experiment_2
from config import batch_size, max_len, hidden_size, num_layers
from utils import compare_models
from models.lstm_model import LSTM
from models.gru_model import GRU

if __name__ == "__main__":
    device = setup_environment()
    corpus_path = 'cleaned_lemonde_corpus.txt'
    tokenizer, vocab = initialize_tokenizer_and_vocab(corpus_path)
    train_loader, test_loader = create_dataset_and_loaders(corpus_path, vocab, tokenizer, max_len=8, batch_size=64)

    # Initialize LSTM model
    lstm_checkpoint_dir = "path/to/lstm/checkpoints"
    lstm_model, lstm_optimizer, start_epoch, train_losses, test_losses, perplexities, loss_fn = initialize_lstm_model(
        vocab, num_layers, hidden_size, checkpoint_dir=lstm_checkpoint_dir, device=device
    )

    # Initialize GRU model
    gru_model, gru_optimizer, gru_loss_fn = initialize_gru_model(vocab, num_layers, hidden_size, device=device)

    # Example: Running Experiment 1
    experiment_1(train_loader, test_loader, vocab, device)
