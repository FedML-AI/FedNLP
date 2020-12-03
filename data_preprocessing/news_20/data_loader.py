import os
import re

from data_preprocessing.base.base_client_data_loader import BaseClientDataLoader
from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader

_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                           r'|^In article|^Quoted from|^\||^>)')


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "text_classification"
        self.train_path = "20news-bydate-train"
        self.test_path = "20news-bydate-test"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X = []
            Y = []
            for root1, dirs, _ in os.walk(os.path.join(self.data_path, self.train_path)):
                for dir in dirs:
                    for root2, _, files in os.walk(os.path.join(root1, dir)):
                        for file_name in files:
                            file_path = os.path.join(root2, file_name)
                            X.extend(self.process_data(file_path))
                            Y.append(dir)
            train_size = len(X)
            for root1, dirs, _ in os.walk(os.path.join(self.data_path, self.test_path)):
                for dir in dirs:
                    for root2, _, files in os.walk(os.path.join(root1, dir)):
                        for file_name in files:
                            file_path = os.path.join(root2, file_name)
                            X.extend(self.process_data(file_path))
                            Y.append(dir)
            self.X, self.Y = X, Y
            train_index_list = [i for i in range(train_size)]
            test_index_list = [i for i in range(train_size, len(X))]
            index_list = train_index_list + test_index_list
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
            self.attributes = {"index_list": index_list, "train_index_list": train_index_list,
                               "test_index_list": test_index_list}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    # remove header
    def remove_headers(self, text):
        _before, _blankline, after = text.partition('\n\n')
        return after

    # remove quotes
    def remove_quotes(self, text):
        good_lines = [line for line in text.split('\n')
                      if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    # remove footers
    def remove_footers(self, text):
        lines = text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break

        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text

    def process_data(self, file_path):
        X = []
        with open(file_path, "r", errors='ignore') as f:
            content = f.read()
            content = self.remove_headers(content)
            content = self.remove_footers(content)
            content = self.remove_quotes(content)

            X.append(content)
        return X


class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ["X", "Y"]
        attribute_fields = ["target_vocab"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields,
                         attribute_fields)
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = [token.text.strip().lower() for token in tokenizer(data["X"][i].strip()) if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)
