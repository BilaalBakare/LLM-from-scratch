import numpy as np

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
        

