import numpy as np
import heapq

class SlidingWindow:
    """Efficient sliding window for score tracking using two heaps for quantile computation."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.window = []
        self.scores = []

    def append(self, score):
        self.window.append(score)
        if len(self.window) > self.capacity:
            removed = self.window.pop(0)
            # This is inefficient for large windows but simple.
            # In a production setting, we'd use a balanced BST or two heaps with lazy removal.
            self.scores.remove(removed)
        
        # Keep scores sorted for quantile
        import bisect
        bisect.insort(self.scores, score)

    def quantile(self, q):
        if not self.scores:
            return float('inf')
        # Standard conformal quantile: ceil((n+1)q)
        n = len(self.scores)
        idx = int(np.ceil((n + 1) * q)) - 1
        idx = min(max(idx, 0), n - 1)
        return self.scores[idx]

    def __len__(self):
        return len(self.scores)

class CUSUM:
    """CUSUM-style fail-safe for sequential score-exceedance detection."""
    def __init__(self, alpha, h):
        self.alpha = alpha
        self.h = h
        self.S = 0.0

    def update(self, score, threshold):
        indicator = 1.0 if score > threshold else 0.0
        self.S = max(0.0, self.S + indicator - self.alpha)
        return self.S > self.h

    def reset(self):
        self.S = 0.0
