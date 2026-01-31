import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device, is_cilp=False):
    """
    Unified training function for all assessment tasks.
    is_cilp=False: Performs binary classification 
    is_cilp=True: Performs contrastive alignment 
    """
    model.train()
    total_loss = 0
    
    # 
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for rgb, lidar, labels in pbar:
        # Move all data to the same device (GPU or CPU)
        rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if is_cilp:
            # Contrastive Logic
            r_emb, l_emb = model(rgb, lidar)
            
            # Compute similarity matrix
            logits = torch.matmul(r_emb, l_emb.T) / 0.07 
            
            targets = torch.arange(r_emb.size(0)).to(device)
            loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            
        else:
            #Classification Logic
            outputs = model(rgb, lidar)
            
            #handle shape mismatch
            if outputs.shape != labels.shape:
                outputs = outputs.view(-1)
                
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)