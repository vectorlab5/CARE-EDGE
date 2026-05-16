import hashlib
import hmac
import json
from typing import List

class ProvenanceTagger:
    def __init__(self, key: bytes):
        self.key = key
        self.buffer = []

    def generate_tag(self, metadata: dict) -> str:
        """
        Generates an HMAC-SHA256 tag for the given metadata.
        metadata expected keys: model_hash, input_hash, prediction, score, threshold, route, state_hash, alpha, W, r, epsilon, t
        """
        # Serialize metadata in a deterministic way
        serialized = json.dumps(metadata, sort_keys=True).encode('utf-8')
        tag = hmac.new(self.key, serialized, hashlib.sha256).hexdigest()
        return tag

    def compute_merkle_root(self, tags: List[str]) -> str:
        """Computes the Merkle root of a list of tags."""
        if not tags:
            return ""
        
        nodes = [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in tags]
        
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    combined = nodes[i] + nodes[i+1]
                else:
                    combined = nodes[i] + nodes[i] # Duplicate if odd
                new_nodes.append(hashlib.sha256(combined.encode('utf-8')).hexdigest())
            nodes = new_nodes
            
        return nodes[0]

    def add_to_buffer(self, tag: str, N: int, N_prime: int):
        """Adds tag to buffer and returns Merkle root if batch is full."""
        self.buffer.append(tag)
        result = {"merkle_root": None, "anchor_sepolia": False}
        
        if len(self.buffer) >= N:
            root = self.compute_merkle_root(self.buffer)
            result["merkle_root"] = root
            self.buffer = [] # Clear buffer
            
            # Logic for N' batches could be added here to trigger Sepolia anchoring
            # For simplicity, we just return the root.
            
        return result
