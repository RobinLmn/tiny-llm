import re
import tiktoken
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs"""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert a list of token IDs to text"""
        pass

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary"""
        pass

class CustomTokenizer(Tokenizer):
    def __init__(self):
        self.vocabulary = {'<UNK>': 0}
        self.inverted_vocabulary = {0: '<UNK>'}
        self.token_id = 1

    def build_vocabulary(self, text: str):
        words = list(set(self._split(text)))
        for word in words:
            if word in self.vocabulary:
                continue
            self.vocabulary[word] = self.token_id
            self.inverted_vocabulary[self.token_id] = word
            self.token_id += 1

    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in self._split(text):
            if not word in self.vocabulary:
                word = '<UNK>'
            tokens.append(self.vocabulary[word])
        return tokens

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)
    
    @abstractmethod
    def _split(self, text: str) -> List[str]:
        """Splits a text into 'words' based on the tokenizer's type"""
        pass

class CharacterTokenizer(CustomTokenizer):
    def _split(self, text: str) -> List[str]:
        return list(text)
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.inverted_vocabulary[token] for token in tokens])
    
class WordTokenizer(CustomTokenizer):
    def _split(self, text: str) -> List[str]:
        return re.findall(r'\w+|[^\w\s]', text.lower())
    
    def decode(self, tokens: List[int]) -> str:
        return ' '.join([self.inverted_vocabulary[token] for token in tokens])

class SubWordTokenizer(Tokenizer):
    def __init__(self, encoding : str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> List[int]:
        return self.encoder.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.encoder.decode(tokens)
    
    @property
    def vocabulary_size(self) -> int:
        return self.encoder.n_vocab
