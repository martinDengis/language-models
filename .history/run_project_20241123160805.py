from setup import setup_environment
from dataset import initialize_tokenizer_and_vocab, create_dataset_and_loaders
from training import train_model, generate_text
from experiments import run_experiments
from config import batch_size, max_len, hidden_size, num_layers
from models.lstm_model import LSTM
from models.gru_model import GRU
from utils import compare_models

setup_environment()

corpus_path = 'cleaned_lemonde_corpus.txt'

# Initialize tokenizer and vocabulary
tokenizer, vocab = initialize_tokenizer_and_vocab(corpus_path)

# Create dataset and data loaders
train_loader, test_loader = create_dataset_and_loaders(corpus_path, vocab, tokenizer, max_len, batch_size)

# Train LSTM and GRU models
lstm_model = LSTM(num_emb=len(vocab), output_size=len(vocab))
gru_model = GRU(num_emb=len(vocab), output_size=len(vocab))

lstm_results = train_model(lstm_model, train_loader, test_loader, vocab, "LSTM")
gru_results = train_model(gru_model, train_loader, test_loader, vocab, "GRU")

# Get model parameters
lstm_params = sum(p.numel() for p in lstm_model.parameters())
gru_params = sum(p.numel() for p in gru_model.parameters())

# Compare models
compare_models(lstm_results, gru_results, lstm_params, gru_params)

# Generate text using the trained models
seed_text = "Le président"
print("Generated text with LSTM:")
print(generate_text(lstm_model, vocab, tokenizer, seed_text))

print("Generated text with GRU:")
print(generate_text(gru_model, vocab, tokenizer, seed_text))

# Run additional experiments
run_experiments(train_loader, test_loader, vocab)