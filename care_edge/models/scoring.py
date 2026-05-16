import torch
import torch.nn.functional as F
import numpy as np

def compute_classification_score(logits):
    """Equation: s_e(x) = -log p(y_hat | x)"""
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    score = -torch.log(conf + 1e-9)
    return pred, score

def compute_entropy_score(logits):
    """Standard softmax entropy for BranchyNet-style cascades."""
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    return entropy

def compute_quantile_regression_score(lower_quantile, median, upper_quantile):
    """
    For regression and traffic anomaly detection:
    Width-normalised distance from the predictive median to the closest quantile boundary.
    """
    dist_to_lower = torch.abs(median - lower_quantile)
    dist_to_upper = torch.abs(median - upper_quantile)
    width = upper_quantile - lower_quantile
    
    score = torch.min(dist_to_lower, dist_to_upper) / (width + 1e-9)
    return score

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error with n_bins reliability bins."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()
