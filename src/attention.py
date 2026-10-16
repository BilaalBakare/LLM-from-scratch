import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_model):
        self.query = np.random.randn(d_model, d_model) * 0.01
        self.key = np.random.randn(d_model, d_model) * 0.01
        self.value = np.random.randn(d_model, d_model) * 0.01

    
