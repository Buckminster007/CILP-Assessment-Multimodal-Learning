
import torch
import torch.nn as nn
import torch.nn.functional as F

# ENCODERS


class Encoder(nn.Module):
    """
    Standard Encoder using MaxPool2d (Baseline).
    Used for Task 3 and Task 5 (CILP Stage 1).
    """
    def __init__(self, in_channels, emb_size=200):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(50, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(100, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Linear(128, emb_size)

    def forward(self, x):
        return self.head(self.features(x))

class EmbedderStrided(nn.Module):
    """
    Encoder using Strided Convolutions.
    Used for Task 4 Ablation.
    """
    def __init__(self, in_channels, emb_size=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(50, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(100, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Linear(128, emb_size)

    def forward(self, x):
        return self.head(self.features(x))



#UNIFIED MODEL 


class MultiTaskModel(nn.Module):
    def __init__(self, mode='intermediate', strategy='concat', use_strided=False):
        super().__init__()
        self.mode = mode
        self.strategy = strategy
        
        # Select Encoder Class
        EncoderClass = EmbedderStrided if use_strided else Encoder
        
        # Task 3 uses 128 dim by default
        self.rgb_encoder = EncoderClass(in_channels=4, emb_size=128)
        self.lidar_encoder = EncoderClass(in_channels=1, emb_size=128)
        
        if mode == 'late':
            fusion_dim = 256
        elif strategy == 'concat':
            fusion_dim = 256
        else:
            fusion_dim = 128 
            
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, rgb, lidar):
        r = self.rgb_encoder(rgb)
        l = self.lidar_encoder(lidar)
        
        if self.mode == 'late':
            combined = torch.cat((r, l), dim=1)
        else:
            if self.strategy == 'concat':
                combined = torch.cat((r, l), dim=1)
            elif self.strategy == 'add':
                combined = r + l
            elif self.strategy == 'prod':
                combined = r * l
            else:
                combined = torch.cat((r, l), dim=1)
                
        return self.classifier(combined)



#CILP MODELS


class CILP_Model(nn.Module):
    """Stage 1: Contrastive Pretraining"""
    def __init__(self, emb_size=200, use_strided=False):
        super().__init__()
        # Allow switching, but default to Encoder (MaxPool) as it is more stable for CILP
        EncoderClass = EmbedderStrided if use_strided else Encoder
        
        self.rgb_encoder = EncoderClass(in_channels=4, emb_size=emb_size)
        self.lidar_encoder = EncoderClass(in_channels=1, emb_size=emb_size)
        
    def forward(self, rgb, lidar):
        r_emb = F.normalize(self.rgb_encoder(rgb), p=2, dim=1)
        l_emb = F.normalize(self.lidar_encoder(lidar), p=2, dim=1)
        return r_emb, l_emb

class Projector(nn.Module):
    """Stage 2: Cross-Modal Projector"""
    def __init__(self, input_dim=200, output_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class RGB2LiDARClassifier(nn.Module):
    """Stage 3: Final Classifier"""
    def __init__(self, rgb_encoder, projector, input_dim=200):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.projector = projector
        
        # Freeze Backbones
        for p in self.rgb_encoder.parameters(): p.requires_grad = False
        for p in self.projector.parameters(): p.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, rgb):
        with torch.no_grad():
            rgb_emb = self.rgb_encoder(rgb)
            proj_emb = self.projector(rgb_emb)
        return self.classifier(proj_emb)
