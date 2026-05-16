import torch
import numpy as np
import time
from care_edge.engine import CareEdgeEngine
from care_edge.models.backbones import VisionBackbone, TrafficBackbone, AudioBackbone
from care_edge.baselines import AlwaysEdge, AlwaysCloud, EntropyThreshold, DRLOffload, BayesVariance
from care_edge.utils.metrics import MetricTracker
from experiments.dataset_factory import get_dataloader, get_benchmarks

def run_experiment(benchmark_name, policy_type='care-edge', shift_mode='medium', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    benchmarks = get_benchmarks()
    config_bench = benchmarks[benchmark_name]
    modality = config_bench['modality']
    num_classes = config_bench['num_classes']
    
    # Initialize models
    if modality == 'vision':
        f_e = VisionBackbone(num_classes)
        f_c = VisionBackbone(num_classes)
        f_e_ensemble = [VisionBackbone(num_classes) for _ in range(5)]
    elif modality == 'traffic':
        f_e = TrafficBackbone()
        f_c = TrafficBackbone()
        f_e_ensemble = [TrafficBackbone() for _ in range(5)]
    elif modality == 'audio':
        f_e = AudioBackbone(num_classes)
        f_c = AudioBackbone(num_classes)
        f_e_ensemble = [AudioBackbone(num_classes) for _ in range(5)]
        
    config_engine = {
        "alpha": 0.1,
        "W": 2048,
        "r": 0.01,
        "epsilon": 8/255,
        "N": 1000
    }
    
    tracker = MetricTracker(alpha=0.1)
    dataloader = get_dataloader(benchmark_name, shift_mode)
    
    # Initialize policies
    if policy_type == 'care-edge':
        engine = CareEdgeEngine(f_e, f_c, config_engine)
    elif policy_type == 'always-edge':
        policy = AlwaysEdge()
    elif policy_type == 'always-cloud':
        policy = AlwaysCloud()
    elif policy_type == 'entropy-thr':
        policy = EntropyThreshold(threshold=0.5)
    elif policy_type == 'drl-offload':
        policy = DRLOffload()
    elif policy_type == 'bayes-var':
        policy = BayesVariance(threshold=0.01)
    
    # Run stream
    for i, (x, y) in enumerate(dataloader):
        if i >= 2000: # Limit for benchmark runs
            break
            
        start_time = time.perf_counter()
        
        if policy_type == 'care-edge':
            step_result = engine.step(x, label=y)
            route = step_result['route']
            y_served = step_result['prediction']
            # Override prediction if routed to cloud to simulate Assumption 2
            if route == 'c':
                y_served = torch.zeros(1, num_classes)
                y_served[0, y] = 10.0
        else:
            route = policy.decide(x, f_e, f_c, model_edge_ensemble=f_e_ensemble)
            if route == 'e':
                y_served, _ = f_e(x)
            else:
                # Force cloud to be correct for mock demonstration
                y_served = torch.zeros(1, num_classes)
                y_served[0, y] = 10.0
        
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1e6
        
        # Track metrics
        probs = torch.softmax(y_served if isinstance(y_served, torch.Tensor) else torch.zeros(1, num_classes), dim=1)
        loss = 1.0 if y_served.argmax() != y else 0.0
        
        tracker.update(loss, route, probs=probs, label=y, prov_latency=latency_us if policy_type == 'care-edge' else 0)
        
    return tracker.get_results()

if __name__ == "__main__":
    benchmark_list = ["METR-LA", "UrbanSound8K", "BDD100K"]
    policies = ["always-edge", "always-cloud", "entropy-thr", "drl-offload", "bayes-var", "care-edge"]
    
    for b in benchmark_list:
        print(f"\n--- Benchmark: {b} ---")
        for p in policies:
            res = run_experiment(b, policy_type=p)
            print(f"Policy: {p:15} | Mis-cov: {res['mis_coverage']:.3f} | Cloud-rate: {res['cloud_rate']:.3f} | ECE: {res.get('ece', 0):.3f}")
