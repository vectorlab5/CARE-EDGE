import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SmartCityDataset(Dataset):
    def __init__(self, benchmark, mode='mild', size=5000):
        self.benchmark = benchmark
        self.mode = mode
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        if self.benchmark == 'BDD100K':
            x = torch.randn(3, 224, 224)
            y = torch.randint(0, 4, (1,)).item()
        elif self.benchmark == 'METR-LA':
            x = torch.randn(12, 1) # 12 time steps
            y = torch.randint(0, 2, (1,)).item()
        elif self.benchmark == 'UrbanSound8K':
            x = torch.randn(1, 128, 128) # Mel-spectrogram
            y = torch.randint(0, 10, (1,)).item()
        else:
            raise ValueError("Unknown benchmark")
            
        # Apply shifts (Hendrycks severity)
        severity = 0
        if self.mode == 'medium':
            severity = 3
        elif self.mode == 'severe':
            severity = 5
            
        if severity > 0:
            noise_level = 0.05 * severity
            x = x + torch.randn_like(x) * noise_level
            
        return x, y

def get_benchmarks():
    return {
        "BDD100K": {"modality": "vision", "num_classes": 4, "input_shape": (3, 224, 224)},
        "METR-LA": {"modality": "traffic", "num_classes": 2, "input_shape": (12, 1)},
        "UrbanSound8K": {"modality": "audio", "num_classes": 10, "input_shape": (1, 128, 128)}
    }

def get_dataloader(benchmark, mode='mild', batch_size=1):
    dataset = SmartCityDataset(benchmark, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
