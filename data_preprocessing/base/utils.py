# Variables
from spacy.lang.cs import Czech
from spacy.lang.ru import Russian
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from spacy.lang.de import German

from ..base.globals import *

import struct
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


def load_word2vec_embedding(path):
    vocab = dict()
    weights = None
    total_num_vectors, vector_len = None, None
    with open(path, "rb") as f:
        c = None

        # read the header
        header = b""
        while c != b"\n":
            c = f.read(1)
            header += c

        total_num_vectors, vector_len = (int(x) for x in header.decode().split())
        weights = np.zeros((total_num_vectors+2, vector_len))

        while len(vocab) < total_num_vectors:

            word = b""
            while True:
                c = f.read(1)
                if c == b" ":
                    break
                word += c

            binary_vector = f.read(FLOAT_SIZE * vector_len)
            vocab[word.decode()] = len(vocab)
            weights[len(vocab)-1] = np.array([struct.unpack_from('f', binary_vector, i)[0]
                            for i in range(0, len(binary_vector), FLOAT_SIZE)])
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    return vocab, weights


def load_glove_embedding(path):
    vocab = dict()
    weights = []
    vector_len = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            temp = line.split(" ")
            vocab[temp[0]] = len(vocab)
            weights.append([float(num) for num in temp[1:]])
            vector_len = len(temp[1:])
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    weights.append([0.0 for _ in range(vector_len)])
    weights.append([0.0 for _ in range(vector_len)])
    return vocab, np.array(weights)
