import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

def setup_environment():

    # Check TensorFlow GPU availability
    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Check PyTorch MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print(f"Using device: {device}")
    return device
