import os
from functools import reduce

from data_preprocessing.base.base_client_data_loader import BaseClientDataLoader
from data_preprocessing.base.base_raw_data_loader import SeqTaggingRawDataLoader


class RawDataLoader(SeqTaggingRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "wnut17train.conll"
        self.test_file_name = "emerging.test.annotated"

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.attributes["label_vocab"] is None:
            train_size = self.process_data_file(os.path.join(self.data_path, self.train_file_name))
            test_size = self.process_data_file(os.path.join(self.data_path, self.test_file_name))
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size + test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(reduce(lambda a,b: a+b, self.Y.values()))}

    def process_data_file(self, file_path):
        single_x = []
        single_y = []
        cnt = 0
        with open(file_path, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split("\t")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0:
                        assert len(self.X) == len(self.Y)
                        idx = len(self.X)
                        self.X[idx] = single_x.copy()
                        self.Y[idx] = single_y.copy()
                        cnt += 1
                    single_x.clear()
                    single_y.clear()
        return cnt


class ClientDataLoader(BaseClientDataLoader):
    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ["X", "Y"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
