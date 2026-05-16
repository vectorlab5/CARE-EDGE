import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MockDataset(Dataset):
    def __init__(self, mode, modality, size=10000):
        self.size = size
        self.modality = modality
        self.mode = mode
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        if self.modality == 'vision':
            x = torch.randn(3, 224, 224)
            y = torch.randint(0, 4, (1,)).item()
        elif self.modality == 'traffic':
            x = torch.randn(12, 1) # 12 time steps
            y = torch.randint(0, 2, (1,)).item()
        elif self.modality == 'audio':
            x = torch.randn(1, 128, 128) # Mel-spectrogram
            y = torch.randint(0, 10, (1,)).item()
        else:
            raise ValueError("Unknown modality")
            
        # Simulate shift/corruption based on mode
        if self.mode == 'medium':
            x += torch.randn_like(x) * 0.1
        elif self.mode == 'severe':
            x += torch.randn_like(x) * 0.3
            
        return x, y

def get_dataloader(modality, mode, batch_size=1):
    dataset = MockDataset(mode, modality)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_benchmarks():
    return {
        "BDD100K": {"modality": "vision", "num_classes": 4},
        "METR-LA": {"modality": "traffic", "num_classes": 2},
        "UrbanSound8K": {"modality": "audio", "num_classes": 10}
    }
