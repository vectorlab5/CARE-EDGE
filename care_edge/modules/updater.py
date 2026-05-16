import numpy as np
from typing import List, Tuple

class ThresholdUpdater:
    def __init__(self, alpha: float, B: float = 1.0, lambda_param: float = 0.5):
        self.alpha = alpha
        self.B = B
        self.lambda_param = lambda_param

    def compute_crc_threshold(self, labelled_window: List[Tuple[float, float, float]], tau_grid: np.ndarray) -> float:
        """
        Computes the Conformal Risk Control (CRC) threshold.
        labelled_window: List of (edge_loss, cloud_loss, score)
        """
        n = len(labelled_window)
        if n == 0:
            return float('inf')

        # Convert to arrays for vectorization
        edge_losses = np.array([item[0] for item in labelled_window])
        cloud_losses = np.array([item[1] for item in labelled_window])
        scores = np.array([item[2] for item in labelled_window])

        target_risk = self.alpha - self.B / (n + 1)
        
        best_tau = -float('inf')
        
        # Grid search over tau
        for tau in tau_grid:
            # L_tau(X, Y) = edge_loss if score <= tau else cloud_loss
            mask = scores <= tau
            served_losses = np.where(mask, edge_losses, cloud_losses)
            empirical_risk = np.mean(served_losses)
            
            if empirical_risk <= target_risk:
                best_tau = max(best_tau, tau)
        
        return best_tau

    def compute_quantile_threshold(self, score_window_values: List[float]) -> float:
        """Computes the 1-alpha quantile of the scores."""
        if not score_window_values:
            return float('inf')
        
        n = len(score_window_values)
        scores = sorted(score_window_values)
        # Equation 66: Quantile_{1-alpha}(C^s union {inf})
        # This is equivalent to taking the (1-alpha) quantile of n+1 points where one is inf
        idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        if idx >= n:
            return float('inf')
        return scores[idx]

    def compute_tightening_margin(self, adv_scores: List[float], clean_scores: List[float]) -> float:
        """Computes gamma_t based on adversarial and clean score quantiles."""
        if not adv_scores or not clean_scores:
            return 0.0
        
        q_adv = np.quantile(adv_scores, 1 - self.alpha)
        q_clean = np.quantile(clean_scores, 1 - self.alpha)
        
        gamma = self.lambda_param * max(0.0, q_adv - q_clean)
        return gamma

    def get_combined_threshold(self, threshold_crc: float, threshold_q: float, gamma: float) -> float:
        """Combines CRC, Quantile and tightening margin (Equation 43)."""
        base_threshold = min(threshold_crc, threshold_q)
        return base_threshold - gamma
