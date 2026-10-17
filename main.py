import numpy as np
from src.tokenizer import Tokenizer
from src.model import TransformerLM

corpus = "hello how are you doing today I am fine thank you"
tokenizer = Tokenizer()
tokenizer.build_vocab(corpus)

prompt = "hello how"
prompt_ids = tokenizer.encode(prompt)

model = TransformerLM(
    vocab_size=len(tokenizer.word_to_id),
    d_model=32,
    max_seq_len=16,
    num_heads=4
)

output_text = model.generate(
    start_tokens=prompt_ids, 
    max_new_tokens=5, 
    id_to_word=tokenizer.id_to_word
)

print(f"Prompt: '{prompt}'")
print(f"Model Output: '{output_text}'")