import torch
import torch.nn.functional as F
import numpy as np
from .models.scoring import compute_entropy_score

class BaselinePolicy:
    def __init__(self, name):
        self.name = name

    def decide(self, x, model_edge, model_cloud, **kwargs):
        raise NotImplementedError

class AlwaysEdge(BaselinePolicy):
    def __init__(self):
        super().__init__("Always-edge")
    def decide(self, x, model_edge, model_cloud, **kwargs):
        return 'e'

class AlwaysCloud(BaselinePolicy):
    def __init__(self):
        super().__init__("Always-cloud")
    def decide(self, x, model_edge, model_cloud, **kwargs):
        return 'c'

class EntropyThreshold(BaselinePolicy):
    def __init__(self, threshold):
        super().__init__("Entropy-threshold")
        self.threshold = threshold
    def decide(self, x, model_edge, model_cloud, **kwargs):
        with torch.no_grad():
            logits, _ = model_edge(x)
        entropy = compute_entropy_score(logits)
        return 'e' if entropy.item() <= self.threshold else 'c'

class DRLOffload(BaselinePolicy):
    """
    Simplified DRL scheduler. 
    In production, this would be a trained MLP/RNN policy.
    Here we simulate the policy trained to minimize latency-energy-accuracy.
    """
    def __init__(self):
        super().__init__("DRL-offload")
    def decide(self, x, model_edge, model_cloud, **kwargs):
        # Simulation: uses a learned heuristic based on input norm and confidence proxy
        with torch.no_grad():
            _, score = model_edge(x)
        # Heuristic: DRL policy typically learns to be more aggressive with edge
        return 'e' if score.item() < 1.5 else 'c'

class BayesVariance(BaselinePolicy):
    """Thresholds the predictive variance of a 5-member deep ensemble."""
    def __init__(self, threshold):
        super().__init__("Bayes-variance")
        self.threshold = threshold
    def decide(self, x, model_edge, model_cloud, model_edge_ensemble=None, **kwargs):
        if model_edge_ensemble is None:
            return 'c'
        # model_edge_ensemble is expected to be a list of 5 models
        logits_list = []
        with torch.no_grad():
            for m in model_edge_ensemble:
                logits, _ = m(x)
                logits_list.append(F.softmax(logits, dim=1))
        
        # Variance of softmax probabilities across ensemble
        all_probs = torch.stack(logits_list) # (5, batch, classes)
        variance = torch.var(all_probs, dim=0).mean()
        return 'e' if variance.item() <= self.threshold else 'c'
