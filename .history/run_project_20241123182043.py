from setup import setup_environment
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from experiments import experiment_1
from config import hidden_size, num_layers
from models import GRUWord2Vec, LSTMOneHot, GRUOneHot

if __name__ == "__main__":
    # Set up the environment and device
    device = setup_environment()

    # Prepare dataset
    corpus_path = 'cleaned_lemonde_corpus.txt'
    tokenizer, vocab = initialize_tokenizer_and_vocab(corpus_path)
    train_loader, test_loader = create_dataset_and_loaders(corpus_path, vocab, tokenizer, batch_size=64)

    # Run Experiment 1
    experiment_1(train_loader, test_loader, vocab, device)

