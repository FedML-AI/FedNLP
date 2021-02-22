from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def transform(self, X, y=None):
        pass