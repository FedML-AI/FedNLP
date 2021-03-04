import os

from data_preprocessing.base.base_client_data_loader import BaseClientDataLoader
from data_preprocessing.base.base_raw_data_loader import Seq2SeqRawDataLoader


class RawDataLoader(Seq2SeqRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.cnn_path = "cnn/stories"
        self.dailymail_path = "dailymail/stories"

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0:
            total_size = 0
            for root, dirs, files in os.walk(os.path.join(self.data_path, self.cnn_path)):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    processed_size = self.process_data_file(file_path)
                    total_size += processed_size
            for root, dirs, files in os.walk(os.path.join(self.data_path, self.dailymail_path)):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    processed_size = self.process_data_file(file_path)
                    total_size += processed_size
            index_list = [i for i in range(total_size)]
            self.attributes["index_list"] = index_list

    def process_data_file(self, file_path):
        cnt = 0
        article_lines = []
        abstract_lines = []
        next_is_highlight = False
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    if line.startswith("@highlight"):
                        next_is_highlight = True
                    elif next_is_highlight:
                        abstract_lines.append(line)
                    else:
                        article_lines.append(line)
        assert len(self.X) == len(self.Y)
        idx = len(self.X)
        self.X[idx] = " ".join(article_lines)
        self.Y[idx] = " ".join(["%s %s %s" % ("<s>", sent, "</s>") for sent in abstract_lines])
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
                data["Y"][i] = [token.text.strip().lower() for token in tokenizer(data["Y"][i].strip()) if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)
