import torch
import numpy as np
from typing import Dict, Any, List
from .utils.stats import SlidingWindow, CUSUM
from .modules.updater import ThresholdUpdater
from .modules.probe import AdversarialProbe
from .modules.provenance import ProvenanceTagger
import hashlib

class CareEdgeEngine:
    def __init__(self, f_e, f_c, config: Dict[str, Any]):
        self.f_e = f_e
        self.f_c = f_c
        self.config = config
        
        # Hyperparameters
        self.alpha = config.get('alpha', 0.1)
        self.epsilon = config.get('epsilon', 8/255)
        self.r = config.get('r', 0.01)
        self.W = config.get('W', 2048)
        self.N = config.get('N', 1000)
        self.h = config.get('h', 0.5)
        self.k = config.get('hmac_key', b'default_secret_key')
        
        # State
        self.score_window = SlidingWindow(self.W)
        self.adv_window = SlidingWindow(self.W)
        self.labelled_window = [] # List of (e_loss, c_loss, score)
        self.threshold = float('inf')
        self.cusum = CUSUM(self.alpha, self.h)
        self.tagger = ProvenanceTagger(self.k)
        self.t = 0
        
        # Modules
        self.updater = ThresholdUpdater(self.alpha)
        self.probe = AdversarialProbe(self.epsilon)
        
        # Grid for CRC
        self.tau_grid = np.linspace(0, 10, 100) # Adjusted range for -log p

    def step(self, x: torch.Tensor, label=None) -> Dict[str, Any]:
        self.t += 1
        
        # 1. Edge prediction and score
        with torch.no_grad():
            y_hat_e, score_e = self.f_e(x)
        
        # 2. CUSUM update
        triggered = self.cusum.update(score_e, self.threshold)
        
        # 3. Routing decision
        if triggered:
            route = 'c'
            # Trigger cool-down logic could be added here
        else:
            route = 'e' if score_e <= self.threshold else 'c'
            
        # 4. Served prediction
        if route == 'e':
            y_served = y_hat_e
        else:
            with torch.no_grad():
                y_served, _ = self.f_c(x)
                
        # 5. Provenance tag
        metadata = {
            "model_hash": hashlib.sha256(str(self.f_e.state_dict()).encode()).hexdigest(),
            "input_hash": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
            "prediction": y_served.item() if isinstance(y_served, torch.Tensor) and y_served.numel() == 1 else str(y_served),
            "score": float(score_e),
            "threshold": self.threshold,
            "route": route,
            "state_hash": hashlib.sha256((str(self.score_window.scores) + str(self.labelled_window)).encode()).hexdigest(),
            "alpha": self.alpha,
            "W": self.W,
            "r": self.r,
            "epsilon": self.epsilon,
            "t": self.t
        }
        tag = self.tagger.generate_tag(metadata)
        prov_result = self.tagger.add_to_buffer(tag, self.N, 100)
        
        # 6. State updates
        self.score_window.append(score_e)
        
        # Adversarial probe
        s_adv = self.probe.sample_probe(self.f_e, x, self.r)
        if s_adv is not None:
            self.adv_window.append(s_adv)
            
        # Labels and CRC update
        if label is not None:
            # Assume we have a loss function defined somewhere
            # For simplicity, using a dummy loss for now
            e_loss = 1.0 if y_hat_e.argmax() != label else 0.0
            with torch.no_grad():
                y_hat_c, _ = self.f_c(x)
                c_loss = 1.0 if y_hat_c.argmax() != label else 0.0
            self.labelled_window.append((e_loss, c_loss, score_e))
            if len(self.labelled_window) > self.W:
                self.labelled_window.pop(0)
        
        # 7. Recompute threshold for next step
        t_crc = self.updater.compute_crc_threshold(self.labelled_window, self.tau_grid)
        t_q = self.updater.compute_quantile_threshold(self.score_window.scores)
        gamma = self.updater.compute_tightening_margin(self.adv_window.scores, self.score_window.scores)
        
        self.threshold = self.updater.get_combined_threshold(t_crc, t_q, gamma)
        
        return {
            "prediction": y_served,
            "route": route,
            "tag": tag,
            "merkle_root": prov_result["merkle_root"]
        }
