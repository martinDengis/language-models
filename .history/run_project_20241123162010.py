from setup import setup_environment
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from training import train_model, generate_text
from experiments import experiment_1, experiment_2
from config import batch_size, max_len, hidden_size, num_layers
from models.lstm_model import LSTM
from models.gru_model import GRU
from utils import compare_models

def main():
    setup_environment()

    corpus_path = 'cleaned_lemonde_corpus.txt'

    # Initialize tokenizer and vocabulary
    tokenizer, vocab = initialize_tokenizer_and_vocab(corpus_path)

    # Create dataset and data loaders
    train_loader, test_loader = create_dataset_and_loaders(corpus_path, vocab, tokenizer, max_len, batch_size)

    # Run Experiment 1
    experiment_1(train_loader, test_loader, vocab)

    # Run Experiment 2
    experiment_2(train_loader, test_loader, vocab)

if __name__ == "__main__":
    main()