import torch
import torch.nn.functional as F
import numpy as np

class AdversarialProbe:
    """Implements online adversarial (FGSM, PGD) and natural corruption probes."""
    def __init__(self, epsilon: float, device: str = 'cpu'):
        self.epsilon = epsilon
        self.device = device

    def pgd_attack(self, model, x, y_target=None, steps=30, alpha=None):
        """
        30-step PGD attack for METR-LA and robust probing.
        alpha: step size, defaults to epsilon / 10
        """
        if alpha is None:
            alpha = self.epsilon / 4.0
            
        x_adv = x.clone().detach().to(self.device).requires_grad_(True)
        
        if y_target is None:
            # Untargeted: maximize loss
            with torch.no_grad():
                logits, _ = model(x_adv)
                y_target = logits.argmax(dim=1)
        
        for _ in range(steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)
            logits, _ = model(_x_adv)
            loss = F.cross_entropy(logits, y_target)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad = _x_adv.grad.data
                x_adv = x_adv + alpha * grad.sign()
                # Project back to L-infinity ball around x
                eta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + eta, 0, 1)
                
        return x_adv.detach()

    def fgsm_perturb(self, model, x, y_target=None):
        """Simple FGSM perturbation."""
        x = x.clone().detach().to(self.device).requires_grad_(True)
        logits, _ = model(x)
        
        if y_target is None:
            y_target = logits.argmax(dim=1)
            
        loss = F.cross_entropy(logits, y_target)
        model.zero_grad()
        loss.backward()
        
        data_grad = x.grad.data
        perturbed_x = x + self.epsilon * data_grad.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        return perturbed_x.detach()

    def natural_corruption(self, x, severity=3):
        """
        Simulates natural corruptions (Severity 3 and 5) 
        inspired by Hendrycks and Dietterich (2019).
        """
        # Placeholder for complex corruptions: using mixed noise/blur simulation
        noise_level = 0.05 * severity
        blur_kernel = int(severity) if severity % 2 != 0 else int(severity + 1)
        
        # Add noise
        x_noisy = x + torch.randn_like(x) * noise_level
        
        # Simple blur simulation (mean pool)
        if blur_kernel > 1:
            padding = blur_kernel // 2
            x_noisy = F.avg_pool2d(x_noisy, kernel_size=blur_kernel, stride=1, padding=padding)
            
        return torch.clamp(x_noisy, 0, 1)

    def sample_probe(self, model, x, r: float):
        """With probability r, returns a perturbed score."""
        if np.random.rand() > r:
            return None
        
        # Mixed family (default): FGSM + Natural
        if np.random.rand() > 0.5:
            x_adv = self.fgsm_perturb(model, x)
        else:
            x_adv = self.natural_corruption(x, severity=3)
            
        with torch.no_grad():
            _, score_adv = model(x_adv)
        return score_adv.item()
