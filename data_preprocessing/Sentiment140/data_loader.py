import csv
import os
import re
import string

from data_preprocessing.base.base_client_data_loader import BaseClientDataLoader
from data_preprocessing.base.base_raw_data_loader import TextClassificationRawDataLoader


class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.test_file_name = "testdata.manual.2009.06.14.csv"
        self.train_file_name = "training.1600000.processed.noemoticon.csv"

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.attributes["label_vocab"] is None:
            train_size = self.process_data_file(os.path.join(self.data_path, self.train_file_name))
            test_size = self.process_data_file(os.path.join(self.data_path, self.test_file_name))
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size+test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"] 
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y.values()))}

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", newline='', encoding='utf-8', errors='ignore') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for line in data:
                assert len(self.X) == len(self.Y)
                idx = len(self.X)
                self.X[idx] = line[5]
                if line[0] == "0":
                    self.Y[idx] = line[0]
                else:
                    self.Y[idx] = "1"
                cnt += 1

        return cnt


class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ["X", "Y"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        self.clean_data()
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        tokenizer = self.spacy_tokenizer.en_tokenizer

        def __tokenize_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = [token.text.strip() for token in tokenizer(data["X"][i].strip()) if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)

    def clean_data(self):
        def __clean_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = self.clean_str(data["X"][i])
        __clean_data(self.train_data)
        __clean_data(self.test_data)

    # def clean_str(self, sentence):
    #     sentence = re.sub(
    #         r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
    #         " ", sentence)
    #     # Eliminating the token if it is a mention
    #     sentence = re.sub("(@[A-Za-z0-9_]+)", "", sentence)
    #     return sentence.lower()
    def clean_str(self, sentence):
        sentence = re.sub(r'\&\w*;', '', sentence)
        sentence = re.sub('@[^\s]+','',sentence)
        sentence = re.sub(r'\$\w*', '', sentence)
        sentence = sentence.lower()
        sentence = re.sub(r'https?:\/\/.*\/\w*', '', sentence)
        sentence = re.sub(r'#\w*', '', sentence)
        sentence = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', sentence)
        sentence = re.sub(r'\b\w{1,2}\b', '', sentence)
        sentence = re.sub(r'\s\s+', ' ', sentence)
        sentence = [char for char in list(sentence) if char not in string.punctuation]
        sentence = ''.join(sentence)
        sentence = sentence.lstrip(' ') 
        return sentence
