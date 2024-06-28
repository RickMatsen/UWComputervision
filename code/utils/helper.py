import torch
import random
import numpy as np

def set_seed(seed):
    # Set the seed for Python's random number generator
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch for both CPU and CUDA operations
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility for operations with non-deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

