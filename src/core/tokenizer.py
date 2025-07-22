import re
from enum import Enum
from typing import List

class TokenizerType(Enum):
    CHARACTER = "character"
    WORD = "word"

class Tokenizer:
    """Tokenizer"""
    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.WORD):
        self.vocabulary = {'<UNK>': 0}
        self.inverted_vocabulary = {0: '<UNK>'}
        self.token_id = 1
        self.tokenizer_type = tokenizer_type

    def build_vocabulary(self, text: str):
        """Build a vocabulary mapping words to token IDs"""
        words = list(set(self._split(text)))
        for word in words:
            if word in self.vocabulary:
                continue
            self.vocabulary[word] = self.token_id
            self.inverted_vocabulary[self.token_id] = word
            self.token_id += 1

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs"""
        tokens = []
        for word in self._split(text):
            if not word in self.vocabulary:
                word = '<UNK>'
            tokens.append(self.vocabulary[word])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convert a list of token IDs to text"""
        separator = ' ' if self.tokenizer_type == TokenizerType.WORD else ''
        return separator.join([self.inverted_vocabulary[token] for token in tokens])

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary"""
        return len(self.vocabulary)
    
    def _split(self, text: str):
        """Splits a text into 'words' based on the tokenizer's type"""
        if self.tokenizer_type == TokenizerType.CHARACTER:
            return list(text)
        elif self.tokenizer_type == TokenizerType.WORD:
            return re.findall(r'\w+|[^\w\s]', text.lower())
        else:
            raise ValueError(f"Unknown tokenizer type {str(self.tokenizer_type)}")
