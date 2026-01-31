import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seeds(seed=51):
    """Crucial for Task 6 Reproducibility points."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Deterministic environment initialized (Seed: {seed})")

def log_similarity_matrix(sim_matrix, epoch):
    """Required for Task 1.3: Visualizing the image-lidar alignment."""
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix.detach().cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"CILP Similarity Matrix - Epoch {epoch}")
    # This helps you verify the 'diagonal' requirement for the assessment
    plt.show()