import torch
import torch.nn.functional as F
import numpy as np

class AdversarialProbe:
    """Implements online adversarial and natural corruption probes."""
    def __init__(self, epsilon: float, device: str = 'cpu'):
        self.epsilon = epsilon
        self.device = device

    def fgsm_perturb(self, model, x, y_target=None):
        """Simple FGSM perturbation."""
        x = x.clone().detach().to(self.device).requires_grad_(True)
        outputs, _ = model(x)
        
        if y_target is None:
            # Untargeted: maximize loss with respect to top-1 prediction
            y_target = outputs.argmax(dim=1)
            
        loss = F.cross_entropy(outputs, y_target)
        model.zero_grad()
        loss.backward()
        
        data_grad = x.grad.data
        perturbed_x = x + self.epsilon * data_grad.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        return perturbed_x.detach()

    def natural_corruption(self, x, corruption_type='gaussian_noise'):
        """Simulates natural corruptions."""
        if corruption_type == 'gaussian_noise':
            noise = torch.randn_like(x) * self.epsilon
            return torch.clamp(x + noise, 0, 1)
        # Add more types if needed (e.g., brightness, contrast)
        return x

    def sample_probe(self, model, x, r: float):
        """With probability r, returns a perturbed score."""
        if np.random.rand() > r:
            return None
        
        # Mix FGSM and natural corruptions as per paper
        if np.random.rand() > 0.5:
            x_adv = self.fgsm_perturb(model, x)
        else:
            x_adv = self.natural_corruption(x)
            
        with torch.no_grad():
            _, score_adv = model(x_adv)
        return score_adv.item()
