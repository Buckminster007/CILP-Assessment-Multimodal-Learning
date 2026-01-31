import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_multimodal_sample(rgb_tensor, lidar_tensor, label=None):
    """
    Handles Task 2 visualization. 
    Expects raw tensors from the data loader.
    """
    # 1. Convert tensors to numpy
    # Change shape from (Channels, Height, Width) to (Height, Width, Channels)
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Remove the channel dimension for the 2D LiDAR heatmap
    lidar = lidar_tensor.squeeze().cpu().numpy() 

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2. Display RGB (using first 3 channels for visualization)
    ax[0].imshow(rgb[:, :, :3]) 
    ax[0].set_title(f"RGB Image\nLabel: {label}")
    ax[0].axis('off')

    # 3. Display LiDAR as a Depth Heatmap
    im = ax[1].imshow(lidar, cmap='magma')
    ax[1].set_title("LiDAR Depth Map")
    ax[1].axis('off')
    plt.colorbar(im, ax=ax[1], label="Distance")
    
    plt.tight_layout()
    return fig

def log_similarity_matrix(sim_matrix, epoch):
    """
    Handles Task 5 visualization for CILP.
    Logs the similarity heatmap to Weights & Biases.
    """
    matrix_np = sim_matrix.cpu().detach().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix_np, cmap='viridis')
    ax.set_title(f"CILP Alignment Matrix (Epoch {epoch})")
    ax.set_xlabel("LiDAR Embeddings")
    ax.set_ylabel("RGB Embeddings")
    plt.colorbar(im)
    
    # Send image to wandb dashboard
    wandb.log({f"Similarity_Matrix_Epoch_{epoch}": wandb.Image(fig)})
    
    plt.close(fig) # Keeps Colab from slowing down