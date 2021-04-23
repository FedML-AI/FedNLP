# TODO: will finish this part ASAP
import logging
import os
import re
import string

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from data_preprocessing.base.base_example import SeqTaggingInputExample
from data_preprocessing.base.base_preprocessor import BasePreprocessor
from data_preprocessing.utils.seq_tagging_utils import convert_examples_to_features

customized_cleaner_dict = {}


class TrivialPreprocessor(BasePreprocessor):
    # Used for models such as LSTM, CNN, etc.
    def __init__(self, **kwargs):
        super(TrivialPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y):
        pass


class TLMPreprocessor(BasePreprocessor):
    # Used for Transformer language models (TLMs) such as BERT, RoBERTa, etc.
    def __init__(self, **kwargs):
        super(TLMPreprocessor, self).__init__(**kwargs)
        self.text_cleaner = customized_cleaner_dict.get(self.args.dataset, None)

    def transform(self, X, y, index_list=None, evaluate=False):
        if index_list is None:
            index_list = [i for i in range(len(X))]

        examples = self.transform_examples(X, y)
        features = self.transform_features(examples, evaluate)

        dataset = None
        
        return examples, features, dataset

    def transform_examples(self, X, y, index_list):
        return None

    def transform_features(self, examples, evaluate=False, no_cache=False):
        return None

