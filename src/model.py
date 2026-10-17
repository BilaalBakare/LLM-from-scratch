import numpy as np
from embeddings import Embeddings
from transformerblock import TransformerBlock

class TransformerLM:
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.embedder = Embeddings(self.vocab_size, self.d_model, self.max_seq_len)
        self.transformer = TransformerBlock(self.d_model, self.num_heads)
        
        self.W_vocab = np.random.randn(self.d_model, self.vocab_size) * 0.01

    