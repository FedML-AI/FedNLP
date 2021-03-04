import csv
import os

from data_preprocessing.base.base_client_data_loader import BaseClientDataLoader
from data_preprocessing.base.base_raw_data_loader import TextClassificationRawDataLoader


class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_path = "train.csv"
        self.test_path = "test.csv"

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.attributes["label_vocab"] is None:
            train_size = self.process_data_file(os.path.join(self.data_path, self.train_path))
            test_size = self.process_data_file(os.path.join(self.data_path, self.test_path))
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y.values()))}
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size+test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for line in data:
                target = line[0]
                source = line[2].replace('\\', '')
                assert len(self.X) == len(self.Y)
                idx = len(self.X)
                self.X[idx] = source
                self.Y[idx] = target
                cnt += 1
        return cnt


class ClientDataLoader(BaseClientDataLoader):
    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ["X", "Y"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = [token.text.strip().lower() for token in tokenizer(data["X"][i].strip()) if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)
