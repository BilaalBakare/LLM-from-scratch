class Tokenizer:
    def __init__(self, words, text_file=None,):
        # self.text_file = text_file

        words = "hello how are you doing today I am fine thank you"
        
        # with open(self.text_file, 'r', encoding='utf-8') as file:
        #     words = file.read().split()

        self.vocab = ["<UNK>"] + sorted(list(set(words)))
        self.vocab_size = len(self.vocab)

        self.word_to_id = {word:i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i:word for i, word in enumerate(self.vocab)}

    def encode(self, text):
        ids = []

        for word in text.split():
            ids.append(self.word_to_id.get(word, self.word_to_id['<UNK>']))

        return ids

    def decode(self, ids):
        words = []

        for id in ids:
            words.append(self.id_to_word[id])

        return " ".join(words)
        
