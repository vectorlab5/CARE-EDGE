import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class VisionBackbone(nn.Module):
    """MobileNetV3-Small with linear classification head."""
    def __init__(self, num_classes=4, int8_mode=False, perfect=False):
        super().__init__()
        self.model = mobilenet_v3_small(num_classes=num_classes)
        self.int8_mode = int8_mode
        self.perfect = perfect
        
    def forward(self, x):
        if self.perfect:
            # Simulate a perfect model (logits that favor the true label)
            # This is a hack for the mock run
            logits = torch.zeros(x.size(0), 4)
            # In a real run, we'd have labels here. 
            # For now, let's just make it very confident in 'something'.
            logits[:, 0] = 10.0 
        else:
            if self.int8_mode:
                # Simulate int8 quantization noise
                x = torch.round(x * 127) / 127
            logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        score = -torch.log(conf + 1e-9)
        return logits, score

class TrafficBackbone(nn.Module):
    """4-layer Transformer Encoder for traffic time series."""
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Quantile heads for conformalized quantile regression
        self.q_low = nn.Linear(d_model, 1)
        self.q_med = nn.Linear(d_model, 1)
        self.q_high = nn.Linear(d_model, 1)
        
        self.head = nn.Linear(d_model, 2) # Binary anomaly detection
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        feat = self.transformer_encoder(x)
        last_feat = feat[:, -1, :]
        
        # Regression outputs
        low = self.q_low(last_feat)
        med = self.q_med(last_feat)
        high = self.q_high(last_feat)
        
        # Score: width-normalised distance from median to closest boundary
        dist_to_low = torch.abs(med - low)
        dist_to_high = torch.abs(med - high)
        width = high - low + 1e-9
        score = torch.min(dist_to_low, dist_to_high) / width
        
        logits = self.head(last_feat)
        return logits, score.squeeze()

class AudioBackbone(nn.Module):
    """Simplified Audio Spectrogram Transformer (AST) head."""
    def __init__(self, num_classes=10, d_model=128, nhead=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(64 * 8 * 8, d_model) # Assuming input is pooled/conv'd to 8x8
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x is Mel-spectrogram (batch, 1, H, W)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1) # Flatten
        x = self.proj(x).unsqueeze(1) # Add seq dimension
        x = self.pos_encoder(x)
        feat = self.transformer_encoder(x)
        logits = self.head(feat[:, 0, :])
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        score = -torch.log(conf + 1e-9)
        return logits, score
