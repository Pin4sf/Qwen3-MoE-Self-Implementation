# src/kv_cache.py
class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None