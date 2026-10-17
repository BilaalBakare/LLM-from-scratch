import numpy as np
from src.embeddings import Embeddings
from src.transformerblock import TransformerBlock

def softmax(x):
    shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TransformerLM:
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.embedder = Embeddings(self.vocab_size, self.d_model, self.max_seq_len)
        self.transformer = TransformerBlock(self.d_model, self.num_heads)
        
        self.W_vocab = np.random.randn(self.d_model, self.vocab_size) * 0.01

    def forward(self, input_ids):
        embeded = self.embedder.forward(input_ids)  
        hidden_states = self.transformer.forward(embeded)  
        logits = hidden_states @ self.W_vocab  
    
        predictions = softmax(logits) 
        
        return predictions

    def generate(self, start_tokens, max_new_tokens, id_to_word, temperature=0.7):
        tokens = list(start_tokens)
        
        for _ in range(max_new_tokens):
            input_ids = np.array([tokens[-self.max_seq_len:]]) 
            probs = self.forward(input_ids)  
            last_probs = probs[0, -1, :]
            
            next_token_id = int(np.argmax(last_probs))
            
            tokens.append(next_token_id)
            
        words = [id_to_word.get(idx, "<UNK>") for idx in tokens]
        return " ".join(words)