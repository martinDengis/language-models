from setup import setup_environment
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from experiments import experiment_1, experiment_2
from config import batch_size, max_len, hidden_size, num_layers, learning_rate, nepochs, dropout
from models.lstm_model import LSTM
from models.gru_model import GRU
from utils import compare_models

if __name__ == "__main__":
    # Set up the environment and device
    device = setup_environment()

    # Prepare dataset
    corpus_path = 'cleaned_lemonde_corpus.txt'
    tokenizer, vocab = initialize_tokenizer_and_vocab(corpus_path)
    train_loader, test_loader = create_dataset_and_loaders(corpus_path, vocab, tokenizer, batch_size)

    # Run Experiment 1
    experiment_1(train_loader, test_loader, vocab, device)

