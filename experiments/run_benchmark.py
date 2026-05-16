import torch
import numpy as np
import time
from care_edge.engine import CareEdgeEngine
from care_edge.models.backbones import VisionBackbone, TrafficBackbone, AudioBackbone
from experiments.dataset_factory import get_dataloader, get_benchmarks

def run_experiment(benchmark_name, benchmark_config, shift_mode='medium', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    modality = benchmark_config['modality']
    num_classes = benchmark_config['num_classes']
    
    # Initialize models
    if modality == 'vision':
        f_e = VisionBackbone(num_classes)
        f_c = VisionBackbone(num_classes) # Using same architecture for mock
    elif modality == 'traffic':
        f_e = TrafficBackbone(num_classes=num_classes)
        f_c = TrafficBackbone(num_classes=num_classes)
    elif modality == 'audio':
        f_e = AudioBackbone(num_classes)
        f_c = AudioBackbone(num_classes)
        
    config = {
        "alpha": 0.1,
        "W": 2048,
        "r": 0.01,
        "epsilon": 8/255,
        "N": 1000
    }
    
    engine = CareEdgeEngine(f_e, f_c, config)
    dataloader = get_dataloader(modality, shift_mode)
    
    results = []
    start_time = time.time()
    
    # Run stream
    for i, (x, y) in enumerate(dataloader):
        if i >= 5000: # Limit for mock run
            break
            
        # Simulate delay: label arrives after 100 steps
        current_label = None
        # In a real implementation, we'd queue labels
        
        # For this mock, we'll give label immediately for CRC calculation
        step_result = engine.step(x, label=y)
        
        # Track metrics
        edge_pred, _ = f_e(x)
        cloud_pred, _ = f_c(x)
        
        served_pred = step_result['prediction']
        loss = 1.0 if served_pred.argmax() != y else 0.0
        
        results.append({
            "loss": loss,
            "route": step_result['route'],
            "tag": step_result['tag']
        })
        
    end_time = time.time()
    
    # Aggregate results
    mis_coverage = np.mean([r['loss'] for r in results])
    cloud_rate = np.mean([1.0 if r['route'] == 'c' else 0.0 for r in results])
    latency = (end_time - start_time) / len(results) * 1000
    
    return {
        "benchmark": benchmark_name,
        "shift": shift_mode,
        "mis_coverage": mis_coverage,
        "cloud_rate": cloud_rate,
        "latency_ms": latency
    }

if __name__ == "__main__":
    benchmarks = get_benchmarks()
    seeds = [42, 1234, 2026, 7, 31337]
    
    for name, config in benchmarks.items():
        print(f"Running benchmark: {name}")
        res = run_experiment(name, config)
        print(f"Result: {res}")
