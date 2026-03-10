"""
Simple BPE Tokenizer Implementation - From Scratch
For learning purposes - minimal but functional
"""

import json
import re
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

    def get_stats(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, word_freqs, pair):
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus, num_merges=None):
        if num_merges is None:
            num_merges = self.vocab_size - 256

        words = re.findall(r"\w+|[^\w\s]", corpus.lower())
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[" ".join(list(word)) + " </w>"] += 1

        for i in range(256):
            self.token_to_id[chr(i)] = i
            self.id_to_token[i] = chr(i)

        next_id = 256

        for _ in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_vocab(word_freqs, best_pair)
            self.merges.append(best_pair)

            new_token = "".join(best_pair)
            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token
            next_id += 1

            if next_id >= self.vocab_size:
                break

    def tokenize(self, text):
        text = text.lower()
        words = re.findall(r"\w+|[^\w\s]", text)

        tokens = []
        for word in words:
            word_tokens = list(word) + ["</w>"]

            while len(word_tokens) > 1:
                pairs = [
                    (word_tokens[i], word_tokens[i + 1])
                    for i in range(len(word_tokens) - 1)
                ]

                pair_ranks = {pair: i for i, pair in enumerate(self.merges)}

                min_pair = None
                min_rank = float("inf")
                for pair in pairs:
                    if pair in pair_ranks and pair_ranks[pair] < min_rank:
                        min_rank = pair_ranks[pair]
                        min_pair = pair

                if min_pair is None:
                    break

                new_word_tokens = []
                i = 0
                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == min_pair
                    ):
                        new_word_tokens.append("".join(min_pair))
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens

            tokens.extend(word_tokens)

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                for char in token:
                    ids.append(self.token_to_id.get(char, 0))
        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(id, "<unk>") for id in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}


class SimpleTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, corpus):
        chars = sorted(set(corpus.lower()))
        for i, char in enumerate(chars):
            self.char_to_id[char] = i
            self.id_to_char[i] = char

        self.char_to_id["<pad>"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "<pad>"

        self.char_to_id["<unk>"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "<unk>"

    def encode(self, text):
        return [self.char_to_id.get(c, self.char_to_id["<unk>"]) for c in text.lower()]

    def decode(self, ids):
        return "".join([self.id_to_char.get(i, "<unk>") for i in ids])

    @property
    def vocab_size_actual(self):
        return len(self.char_to_id)


if __name__ == "__main__":
    corpus = "Hello world! This is a test corpus for training our tokenizer."

    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(corpus, num_merges=100)

    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Decoded: {decoded}")
