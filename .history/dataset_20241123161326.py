import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator

# Custom Dataset for LeMonde corpus
class LemondeDataset(Dataset):
    """
    A custom dataset class for processing text data from a given file, tokenizing it, and creating sequences for language modeling tasks.

    Attributes:
        sequence_length (int): The length of each sequence.
        data (torch.Tensor): The tensor containing token indices.
        sequences (list): A list of sequences created from the token indices.
        targets (list): A list of target sequences corresponding to the input sequences.

    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Returns the sequence and target at the specified index.

    Args:
        file_path (str): The path to the text file containing the corpus.
        vocab (dict): A dictionary mapping tokens to their corresponding indices.
        tokenizer (callable): A function or callable object that tokenizes the text.
        sequence_length (int): The length of each sequence to be created.
    """
    def __init__(self, file_path, vocab, tokenizer, sequence_length):
        self.sequence_length = sequence_length

        # Read the corpus
        print("Reading corpus from:", file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Corpus length: {len(text)} characters")

        # Tokenize the entire text
        print("Tokenizing text...")
        tokens = tokenizer(text)
        print(f"Number of tokens: {len(tokens)}")

        # Convert tokens to indices
        print("Converting tokens to indices...")
        self.data = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

        # Create sequences and targets
        print("Creating sequences...")
        self.sequences = []
        self.targets = []

        for i in range(0, len(self.data) - sequence_length):
            sequence = self.data[i:i + sequence_length]
            target = self.data[i + 1:i + sequence_length + 1]
            self.sequences.append(sequence)
            self.targets.append(target)

        print(f"Created {len(self.sequences)} sequences")

    def __len__(self):
        """Returns the number of sequences in the object."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Retrieve the sequence and target at the specified index."""
        return self.sequences[idx], self.targets[idx]

def create_french_tokenizer():
    """Create and return a French tokenizer function"""
    try:
        # Load spaCy model with disabled components for speed
        nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])

        def tokenize_text(text, chunk_size=900000):
            tokens = []
            # Split text into chunks
            chunks = [text[i:i + chunk_size]
                     for i in range(0, len(text), chunk_size)]

            print(f"Processing {len(chunks)} chunks...")

            # Process each chunk
            for chunk in tqdm(chunks):
                doc = nlp(chunk)
                chunk_tokens = [token.text for token in doc]
                tokens.extend(chunk_tokens)

            return tokens

        return tokenize_text

    except OSError:
        print("Installing French language model...")
        import os
        os.system('python -m spacy download fr_core_news_sm')
        nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
        return create_french_tokenizer()

def initialize_tokenizer_and_vocab(corpus_path):
    print("Initializing tokenizer...")
    tokenizer = create_french_tokenizer()

    print(f"Reading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.read()

    def yield_tokens(text):
        yield tokenizer(text)

    print("Building vocabulary...")
    vocab = build_vocab_from_iterator(
        yield_tokens(corpus),
        min_freq=2,
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )
    vocab.set_default_index(vocab['<unk>'])
    print(f"Vocabulary size: {len(vocab)}")
    return tokenizer, vocab

def create_dataset_and_loaders(corpus_path, vocab, tokenizer, sequence_length, batch_size):
    print("Creating dataset...")
    dataset = LemondeDataset(corpus_path, vocab, tokenizer, sequence_length)

    print("Splitting dataset...")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader