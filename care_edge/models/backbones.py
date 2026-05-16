import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F

class VisionBackbone(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = mobilenet_v3_small(num_classes=num_classes)
        
    def forward(self, x):
        logits = self.model(x)
        # Score: -log p(y_hat | x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        score = -torch.log(conf + 1e-9)
        return pred, score

class TrafficBackbone(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.head = nn.Linear(d_model, 2) # 2-class anomaly detection
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x).transpose(0, 1) # (seq_len, batch, d_model)
        feat = self.transformer_encoder(x)
        logits = self.head(feat[-1])
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        score = -torch.log(conf + 1e-9)
        return pred, score

class AudioBackbone(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Simplified Audio Spectrogram Transformer (AST) style head
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.head = nn.Linear(32 * 8 * 8, num_classes)
        
    def forward(self, x):
        # x is Mel-spectrogram
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.head(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        score = -torch.log(conf + 1e-9)
        return pred, score
