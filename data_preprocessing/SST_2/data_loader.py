import os

from ..base.base_client_data_loader import BaseClientDataLoader
from ..base.base_raw_data_loader import BaseRawDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "text_classification"
        self.target_vocab = None
        self.label_file_name = "sentiment_labels.txt"
        self.data_file_name = "dictionary.txt"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            index_list = [i for i in range(len(self.X))]
            self.attributes = {"index_list": index_list}
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type,
                "attributes": self.attributes}

    # def label_level(self, label):
    #     label = float(label)
    #     if label >= 0.0 and label <= 0.2:
    #         return "very negative"
    #     elif label > 0.2 and label <= 0.4:
    #         return "negative"
    #     elif label > 0.4 and label <= 0.6:
    #         return "neutral"
    #     elif label > 0.6 and label <= 0.8:
    #         return "positive"
    #     else:
    #         return "very positive"

    def label_level(self, label):
        label = float(label)
        if label < 0.5:
            return "negative"
        else:
            return "positive"

    def process_data(self, file_path):
        X = []
        Y = []
        label_dict = dict()
        with open(os.path.join(file_path, self.label_file_name)) as f:
            for label_line in f:
                label = label_line.split('|')
                label_dict[label[0].strip()] = label[1]

        with open(os.path.join(file_path, self.data_file_name)) as f:
            for data_line in f:
                data = data_line.strip().split("|")
                X.append(data[0].strip())
                Y.append(self.label_level(label_dict[data[1].strip()]))
        return X, Y


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
