from setup import setup_environment
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from experiments import experiment_1, experiment_2

if __name__ == "__main__":
    device = setup_environment()  # Get the device
    corpus_path = 'cleaned_lemonde_corpus.txt'
    tokenizer, vocab = initialize_tokenizer_and_vocab(corpus_path)
    train_loader, test_loader = create_dataset_and_loaders(corpus_path, vocab, tokenizer, max_len=8, batch_size=64)

    # Tokenized corpus for Word2Vec training
    tokenized_corpus = [list(vocab.keys())]  # Replace with actual tokenized data

    # Run experiments
    experiment_1(train_loader, test_loader, vocab, device)  # Pass device
    experiment_2(train_loader, test_loader, vocab, tokenized_corpus, device)  # Pass device
