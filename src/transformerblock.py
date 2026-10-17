import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        return self.gamma * x_norm + self.beta

class ScaledDotProductAttention:
    def __init__(self, d_model):
        self.d_model = d_model

        self.query = np.random.randn(self.d_model, self.d_model) * 0.01
        self.key = np.random.randn(self.d_model, self.d_model) * 0.01
        self.value = np.random.randn(self.d_model, self.d_model) * 0.01

    def forward(self, input):
        B, T, _ = input.shape
        
        Q = input @ self.query
        K = input @ self.key
        V = input @ self.value
        
        K_T = K.transpose(0, 2, 1)
        scores = (Q @ K_T) / np.sqrt(self.d_model)
        
        mask = np.triu(np.ones((T, T)), k=1)
        scores = np.where(mask == 1, -1e9, scores)
        
        weights = self._softmax(scores) 
        
        return weights @ V 

    def _softmax(self, scores):
        shift = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shift)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projection weights
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        # Core attention engine sub-module
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        B, T, _ = x.shape
        
        Q = x @ self.W_q  # (B, T, d_model)
        K = x @ self.W_k  # (B, T, d_model)
        V = x @ self.W_v  # (B, T, d_model)
        
        Q = Q.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        context = self.attention.forward(Q, K, V)  # Shape: (B, h, T, d_k)

        context = context.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        
        out = context @ self.W_o
        
        return out
