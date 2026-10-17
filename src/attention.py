import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_model):
        self.d_model = d_model

        self.query = np.random.randn(self.d_model, self.d_model) * 0.01
        self.key = np.random.randn(self.d_model, self.d_model) * 0.01
        self.value = np.random.randn(self.d_model, self.d_model) * 0.01

    def forward(self, input):
        Q = input @ self.query
        K = input @ self.key
        V = input @ self.value

        K_T = K.transpose(0, 2, 1)
        scores = (Q @ K_T) / np.sqrt(self.d_model)
        causal_mask = np.tril(np.ones((self.d_model, self.d_model), dtype=bool))

        scores += causal_mask

        weights = self._softmax(scores)

        return weights @ V

    def _softmax(self, scores):
        shift = scores - np.max(scores)

        return np.exp(shift) / np.sum(np.exp(shift))
        

