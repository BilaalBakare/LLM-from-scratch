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

        scores = (Q @ np.matrix_transpose(K)) / np.sqrt(self.d_model)

        weights = self._softmax(scores)

        return weights @

    def _softmax(self, scores):
        shift = scores - np.max(scores)

        return np.exp(shift) / np.sum(np.exp(shift))
        

