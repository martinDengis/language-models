import time
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def load_corpus(file_path):
    """
    Loads the text corpus from a specified file.

    Args:
        file_path (str): The path to the file containing the text corpus.

    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        logging.info("File read successfully")
        return content

def create_input_sequences(corpus, tokenizer):
    """
    Generates tri-gram input sequences from a given text corpus using a tokenizer.

    Args:
        corpus (str): The text corpus to be tokenized and converted into sequences.
        tokenizer (Tokenizer): An instance of a tokenizer used to convert text into sequences of tokens.

    Returns:
        list: A list of input sequences, where each sequence is a list of token indices.
    """
    input_sequences = []

    # Convert the entire corpus into a sequence of token indices
    token_list = tokenizer.texts_to_sequences([corpus])[0]

    for i in range(2, len(token_list)):
        # Create tri-gram sequences from the token list.
            # tri-gram sequence: sequence of 3 tokens where the last token is the label to be predicted.
        tri_gram_sequence = token_list[i-2:i+1]
        input_sequences.append(tri_gram_sequence)

    logging.info("Tri-gram input sequences created")
    return input_sequences

def split_sequences(input_sequences, total_words):
    """
    Splits fixed-size input sequences into predictors and labels.

    Args:
        input_sequences (list of list of int): A list of sequences, where each sequence is a list of integers.
        total_words (int): The total number of unique words in the vocabulary.

    Returns:
        tuple: A tuple containing:
            - predictors (numpy.ndarray): The input sequences excluding the last element of each sequence.
            - label (numpy.ndarray): The one-hot encoded labels corresponding to the last element of each sequence.
    """
    input_sequences = np.array(input_sequences)
    predictors = input_sequences[:, :-1]  # all elements except the last
    label = input_sequences[:, -1]  # last element

    # one-hot encoding: transforms the integer labels into binary vectors with one element set to 1 and all others set to 0.
    label = tf.keras.utils.to_categorical(label, num_classes=total_words)
    logging.info("Sequences split into predictors and labels")
    return predictors, label

def create_model(total_words, max_sequence_len):
    """
    Creates and compiles an LSTM-based neural network model for text generation.

    Args:
        total_words (int): The size of the vocabulary.
        max_sequence_len (int): The maximum length of input sequences.

    Returns:
        keras.models.Sequential: A compiled Keras Sequential model.
    """
    logging.info("Creating model")

    # Initialize the Sequential model
    model = Sequential()
    logging.info("- Sequential model created")

    # Add an Embedding layer to learn word embeddings
        # Embedding layer: used to convert the integer indices to dense vectors of fixed size.
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    logging.info("- Embedding layer added")

    # Add an LSTM layer with 150 units followed by another LSTM layer with 100 units
        # LSTM Layer: layer of LSTM units that will learn the patterns in the input sequences.
    model.add(LSTM(150, return_sequences=True))
    logging.info("- First LSTM layer added")
    model.add(LSTM(100))
    logging.info("- Second LSTM layer added")

    # Add a Dense layer with softmax activation for output
        # Dense Layer: fully connected layer that will produce the output predictions.
    model.add(Dense(total_words, activation='softmax'))
    logging.info("- Dense layer added")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info("- Model compiled")

    # Return the compiled model
    logging.info("----------\nModel creation success\n")
    return model

def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    """
    Generates text by predicting the next words based on a seed text using a trained model.

    Args:
        seed_text (str): The initial text to start the generation from.
        next_words (int): The number of words to generate.
        model (keras.Model): The trained language model used for prediction.
        max_sequence_len (int): The maximum length of the sequence used for padding.
        tokenizer (keras.preprocessing.text.Tokenizer): The tokenizer used to convert text to sequences.

    Returns:
        str: The generated text after appending the predicted words to the seed text.
    """
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

if __name__ == '__main__':
    # Limit TensorFlow memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.error(e)

    # List available physical devices
    physical_devices = tf.config.list_physical_devices('GPU')
    logging.info("Num GPUs Available: %d", len(physical_devices))
    logging.info("Available GPUs: %s", physical_devices)

    corpus = load_corpus('lemonde_corpus.txt')

    # Tokenise corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus])
    total_words = len(tokenizer.word_index) + 1
    logging.info('Corpus tokenized. Total words: %d', total_words)

    # Create input sequences
    input_sequences = create_input_sequences(corpus, tokenizer)

    # Pad sequences and split data
    predictors, label = split_sequences(input_sequences, total_words)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(predictors, label, test_size=0.3, random_state=42)  # 70% training, 30% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # from the 30% test -> 15% validation, 15% test
    logging.info("Datasets created")

    # Create the model
    model = create_model(total_words, 3)
    logging.info("----------")
    model.summary(print_fn=logging.info)
    logging.info("----------")

    # Train the model
        # Epochs: number of times the model will cycle through the data.
        # Verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch.
    logging.info("Training model")
        # Create an EarlyStopping callback instance
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )
        # Train the model with the EarlyStopping callback
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=[early_stopping]
    )
    logging.info("Training completed in %.2f seconds", time.time() - start_time)

    # Evaluate the model
    logging.info("Evaluating model")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    perplexity = np.exp(loss)   # Perplexity: measure of how well a probability model predicts a sample.
    logging.info('- Test Accuracy: %.4f', accuracy)
    logging.info('- Test Perplexity: %.4f', perplexity)

    generated_text = generate_text("Le président", 20, model, 3, tokenizer)
    logging.info("Generated text: %s", generated_text)
    