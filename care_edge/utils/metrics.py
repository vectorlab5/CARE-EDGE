import numpy as np
import torch
from ..models.scoring import compute_ece

class MetricTracker:
    def __init__(self, alpha=0.1, cc_ce_ratio=50.0):
        self.alpha = alpha
        self.cc_ce_ratio = cc_ce_ratio
        self.reset()

    def reset(self):
        self.losses = []
        self.routes = []
        self.all_probs = []
        self.all_labels = []
        self.prov_latencies = []
        self.tag_sizes = []

    def update(self, loss, route, probs=None, label=None, prov_latency=0, tag_size=0):
        self.losses.append(loss)
        self.routes.append(1.0 if route == 'c' else 0.0)
        if probs is not None and label is not None:
            self.all_probs.append(probs)
            self.all_labels.append(label)
        if prov_latency > 0:
            self.prov_latencies.append(prov_latency)
        if tag_size > 0:
            self.tag_sizes.append(tag_size)

    def compute_reward(self, accuracy, latency_ms, energy_mj):
        """
        Calculates the DRL reward as a weighted sum of accuracy, latency, and energy.
        He et al. (2024) style reward function.
        """
        # Weights (normalized)
        w_acc, w_lat, w_en = 1.0, 0.1, 0.05
        reward = w_acc * accuracy - (w_lat * latency_ms + w_en * energy_mj)
        return reward

    def get_results(self):
        cloud_rate = np.mean(self.routes) if self.routes else 0
        mis_coverage = np.mean(self.losses) if self.losses else 0 # Simple loss fraction
        
        # Expected Cost: 1 + rho(cc/ce - 1)
        expected_cost = 1.0 + cloud_rate * (self.cc_ce_ratio - 1.0)
        
        results = {
            "mis_coverage": mis_coverage,
            "cloud_rate": cloud_rate,
            "expected_cost": expected_cost,
        }
        
        if self.all_probs and self.all_labels:
            all_p = torch.cat(self.all_probs)
            all_l = torch.tensor(self.all_labels)
            results["ece"] = compute_ece(all_p, all_l)
            
        if self.prov_latencies:
            results["prov_latency_us"] = np.mean(self.prov_latencies)
            
        if self.tag_sizes:
            results["tag_size_bytes"] = np.mean(self.tag_sizes)
            
        return results
