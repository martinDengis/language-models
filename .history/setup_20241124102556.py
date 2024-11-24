import torch

def setup_environment():

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
