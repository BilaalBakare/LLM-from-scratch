import numpy as np

class Embeddings:
    def __init__(self, vocab_size, d_model, max_seq_len):
        W_tok = np.random.randn(vocab_size, d_model) * 0.01
        W_pos = np.random.randn(max_seq_len, d_model) * 0.01

    def forward(self, idx):
        