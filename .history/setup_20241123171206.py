import torch
import torchtext
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import spacy

def setup_environment():
    import torch
    import spacy
    import tensorflow as tf

    # Load the French language model for spaCy
    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        print("Downloading spaCy French language model...")
        spacy.cli.download("fr_core_news_sm")
        nlp = spacy.load("fr_core_news_sm")

    # Check TensorFlow GPU availability
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Check PyTorch MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device available")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU")

    print(f"Using device: {device}")
    return device
