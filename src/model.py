import numpy as np
from tokenizer import Tokenizer
from embeddings import Embeddings
from transformerblock import TransformerBlock

class TransformerLM:
    def __init__(self, input_ids, vocab_size, d_model, max_seq_len, num_heads):
        self.input = input_ids
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads


        self.embedder = Embeddings(self.vocab_size, self.d_model, self.max_seq_len)
        self.transformer = TransformerBlock(self.dmodel, self.num_heads)

    def f