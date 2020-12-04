# Variables
from spacy.lang.cs import Czech
from spacy.lang.ru import Russian
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from spacy.lang.de import German

from ..base.globals import *

import gensim
import numpy as np

FLOAT_SIZE = 4

class SpacyTokenizer:
    def __init__(self):
        self.__zh_tokenizer = None
        self.__en_tokenizer = None
        self.__cs_tokenizer = None
        self.__de_tokenizer = None
        self.__ru_tokenizer = None

    @staticmethod
    def get_tokenizer(lang):
        if lang == "zh":
            # nlp = spacy.load("zh_core_web_sm")
            nlp = Chinese()
        elif lang == "en":
            # nlp = spacy.load("en_core_web_sm")
            nlp = English()
        elif lang == "cs":
            nlp = Czech()
        elif lang == "de":
            # nlp = spacy.load("de_core_web_sm")
            nlp = German()
        elif lang == "ru":
            nlp = Russian()
        else:
            raise Exception("Unacceptable language.")
        return nlp

    @property
    def zh_tokenizer(self):
        if self.__zh_tokenizer is None:
            self.__zh_tokenizer = self.get_tokenizer("zh")
        return self.__zh_tokenizer

    @property
    def en_tokenizer(self):
        if self.__en_tokenizer is None:
            self.__en_tokenizer = self.get_tokenizer("en")
        return self.__en_tokenizer

    @property
    def cs_tokenizer(self):
        if self.__cs_tokenizer is None:
            self.__cs_tokenizer = self.get_tokenizer("cs")
        return self.__cs_tokenizer

    @property
    def de_tokenizer(self):
        if self.__de_tokenizer is None:
            self.__de_tokenizer = self.get_tokenizer("de")
        return self.__de_tokenizer

    @property
    def ru_tokenizer(self):
        if self.__ru_tokenizer is None:
            self.__ru_tokenizer = self.get_tokenizer("ru")
        return self.__ru_tokenizer


def build_vocab(x):
    # x -> [num_seqs, num_tokens]
    vocab = dict()
    for single_x in x:
        for token in single_x:
            if token not in vocab:
                vocab[token] = len(vocab)
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    return vocab


def padding_data(x, max_sequence_length):
    padding_x = []
    for single_x in x:
        new_single_x = single_x.copy()
        if len(new_single_x) <= max_sequence_length:
            for _ in range(len(new_single_x), max_sequence_length):
                new_single_x.append(PAD_TOKEN)
        else:
            new_single_x = new_single_x[:max_sequence_length]
        padding_x.append(new_single_x)
    return padding_x


def token_to_idx(x, vocab):
    idx_x = []
    for single_x in x:
        new_single_x = []
        for token in single_x:
            if token in vocab:
                new_single_x.append(vocab[token])
            else:
                new_single_x.append(vocab[UNK_TOKEN])
        idx_x.append(new_single_x)
    return idx_x


def label_to_idx(y, vocab):
    idx_y = []
    for label in y:
        idx_y.append(vocab[label])
    return idx_y


def load_embedding(path, binary, source_vocab):
    vocab = dict()
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)
    weights = []
    for key, value in model.vocab.items():
        if key in source_vocab:
            vocab[key] = len(vocab)
            weights.append(model.vectors[value.index])
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    weights.append(np.zeros(model.vector_size))
    weights.append(np.zeros(model.vector_size))
    weights = np.array(weights)
    return vocab, weights


