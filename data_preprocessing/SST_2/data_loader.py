import os

from data_preprocessing.base.base_client_data_loader import BaseClientDataLoader
from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


class RawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "text_classification"
        self.target_vocab = None
        self.label_file_name = "sentiment_labels.txt"
        self.dictionary_file_name = "dictionary.txt"
        self.sentence_file_name = "datasetSentences.txt"
        self.split_file_name = "datasetSplit.txt"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y, train_index_list, test_index_list = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            index_list = [i for i in range(len(X))]
            self.attributes = {"index_list": index_list, "train_index_list": train_index_list,
                               "test_index_list": test_index_list}
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
    
    def clean_sentence_string(self, sentence):
        sentence = sentence.replace('-LRB-', '(')
        sentence = sentence.replace('-RRB-', ')')
        sentence = sentence.replace('Â', '')
        sentence = sentence.replace('Ã©', 'e')
        sentence = sentence.replace('Ã¨', 'e')
        sentence = sentence.replace('Ã¯', 'i')
        sentence = sentence.replace('Ã³', 'o')
        sentence = sentence.replace('Ã´', 'o')
        sentence = sentence.replace('Ã¶', 'o')
        sentence = sentence.replace('Ã±', 'n')
        sentence = sentence.replace('Ã¡', 'a')
        sentence = sentence.replace('Ã¢', 'a')
        sentence = sentence.replace('Ã£', 'a')
        sentence = sentence.replace('\xc3\x83\xc2\xa0', 'a')
        sentence = sentence.replace('Ã¼', 'u')
        sentence = sentence.replace('Ã»', 'u')
        sentence = sentence.replace('Ã§', 'c')
        sentence = sentence.replace('Ã¦', 'ae')
        sentence = sentence.replace('Ã­', 'i')
        sentence = sentence.replace('Ã ', 'a')
        sentence = sentence.replace('\xa0', ' ')
        sentence = sentence.replace('\xc2', '')
        return sentence
    
    def clean_dictionary_string(self, sentence):
        sentence = sentence.replace('é', 'e')
        sentence = sentence.replace('è', 'e')
        sentence = sentence.replace('ï', 'i')
        sentence = sentence.replace('í', 'i')
        sentence = sentence.replace('ó', 'o')
        sentence = sentence.replace('ô', 'o')
        sentence = sentence.replace('ö', 'o')
        sentence = sentence.replace('á', 'a')
        sentence = sentence.replace('â', 'a')
        sentence = sentence.replace('ã', 'a')
        sentence = sentence.replace('à', 'a')
        sentence = sentence.replace('ü', 'u')
        sentence = sentence.replace('û', 'u')
        sentence = sentence.replace('ñ', 'n')
        sentence = sentence.replace('ç', 'c')
        sentence = sentence.replace('æ', 'ae')
        sentence = sentence.replace('\xa0', ' ')
        sentence = sentence.replace('\xc2', '')
        return sentence

    def process_data(self, file_path):
        X = []
        label_dict = dict()
        sentence_label_dict = dict()
        train_index_list = []
        test_index_list = []
        with open(os.path.join(file_path, self.split_file_name), "r") as f:
            for i, line in enumerate(f):
                if i != 0:
                    line = line.strip()
                    sent_idx, split_label = line.split(",")
                    if split_label == "1":
                        train_index_list.append(int(sent_idx) - 1)
                    elif split_label == "2":
                        test_index_list.append(int(sent_idx) - 1)
        with open(os.path.join(file_path, self.sentence_file_name), "r") as f:
            for i, sentence_line in enumerate(f):
                if i != 0:
                    sentence_line = sentence_line.strip()
                    sent_idx, sent = sentence_line.split("\t")
                    sent = self.clean_sentence_string(sent)
                    X.append(sent)

        with open(os.path.join(file_path, self.dictionary_file_name)) as f:
            for dictionary_line in f:
                temp = dictionary_line.strip().split("|")
                sent = "|".join(temp[:-1])
                sent = self.clean_dictionary_string(sent)
                if sent in X:
                    sentence_label_dict[sent] = temp[-1]

        with open(os.path.join(file_path, self.label_file_name)) as f:
            for i, label_line in enumerate(f):
                if i != 0:
                    idx, label = label_line.strip().split("|")
                    label_dict[idx] = label
        for sent, label_idx in sentence_label_dict.items():
            sentence_label_dict[sent] = self.label_level(label_dict[label_idx])
        Y = [sentence_label_dict[sent] for sent in X]
        return X, Y, train_index_list, test_index_list


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
                data["X"][i] = [token.text.strip() for token in tokenizer(data["X"][i].lower().strip())
                                if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)
