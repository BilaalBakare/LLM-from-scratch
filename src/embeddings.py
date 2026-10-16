import numpy as np

class Embeddings:
    def __init__(self, vocab_size, d_model, max_seq_len):
        self.W_tok = np.random.randn(vocab_size, d_model) * 0.01
        self.W_pos = np.random.randn(max_seq_len, d_model) * 0.01

    def forward(self, idx):
        self.last_idx = idx # Save input for backward pass

        tok_emb = self.W_tok[idx]

        B, T = idx.shape
        positons = np.arange(T)
        pos_emb = self.W_pos[positons]

        Embedding = tok_emb + pos_emb

        return Embedding