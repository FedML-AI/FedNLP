import json
import os
import random
import re
import nltk
import h5py


from data_preprocessing.base.base_raw_data_loader import SpanExtractionRawDataLoader


class RawDataLoader(SpanExtractionRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "train-v1.1.json"
        self.test_file_name = "dev-v1.1.json"
        self.question_ids = dict()

    def load_data(self):
        if len(self.context_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
            self.attributes["doc_index"] = dict()
            train_size = self.process_data_file(os.path.join(self.data_path, self.train_file_name))
            test_size = self.process_data_file(os.path.join(self.data_path, self.test_file_name))
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size + test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

            for doc_idx, document in enumerate(data["data"]):
                for paragraph in document["paragraphs"]:
                    for qas in paragraph["qas"]:
                        for answer in qas["answers"]:
                            assert len(self.context_X) == len(self.question_X) == len(self.Y) == len(self.question_ids)
                            idx = len(self.context_X)
                            self.context_X[idx] = paragraph["context"]
                            self.question_X[idx] = qas["question"]
                            start = answer["answer_start"]
                            end = start + len(answer["text"].rstrip())
                            self.Y[idx] = (start, end)
                            self.question_ids[idx] = qas["id"]
                            self.attributes["doc_index"][idx] = doc_idx
                            cnt += 1

        return cnt

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.context_X.keys():
            f["context_X/" + str(key)] = self.context_X[key]
            f["question_X/" + str(key)] = self.question_X[key]
            f["Y/" + str(key)] = self.Y[key]
            f["question_ids/" + str(key)] = self.question_ids[key]
        f.close()

class ClientDataLoader(BaseClientDataLoader):


    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False, data_filter=None):
        data_fields = ["context_X", "question_X", "Y", "question_ids"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        self.clean_data()
        if self.tokenize:
            self.tokenize_data()
            self.transform_labels()

        if data_filter:
            data_filter(self.train_data)
            data_filter(self.test_data)

    def clean_data(self):
        def __clean_data(data):
            for i in range(len(data["context_X"])):
                data["context_X"][i] = data["context_X"][i].replace("''", '" ').replace("``", '" ')
        __clean_data(self.train_data)
        __clean_data(self.test_data)

    def tokenize_data(self):

        def word_tokenize(sent):
             return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(sent)]
        def __tokenize_data(data):
            data["tokenized_context_X"] = list()
            data["tokenized_question_X"] = list()
            data["char_context_X"] = list()
            data["char_question_X"] = list()
            self.data_fields.extend(["tokenized_context_X", "tokenized_question_X", "char_context_X", "char_question_X"])
            for i in range(len(data["context_X"])):
                temp_tokens = word_tokenize(data["context_X"][i])
                data["tokenized_context_X"].append(self.remove_stop_tokens(temp_tokens))
                data["tokenized_question_X"].append(word_tokenize(data["question_X"][i]))
                context_chars = [list(token) for token in data["tokenized_context_X"][i]]
                question_chars = [list(token) for token in data["tokenized_question_X"][i]]
                data["char_context_X"].append(context_chars)
                data["char_question_X"].append(question_chars)

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)

    def remove_stop_tokens(self, temp_tokens):
        tokens = []
        for token in temp_tokens:
            flag = False
            l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
            tokens.extend(re.split("([{}])".format("".join(l)), token))
        return tokens

    def transform_labels(self):
        def __transform_labels(data):
            for i in range(len(data["context_X"])):
                context = data["context_X"][i]
                context_tokens = data["tokenized_context_X"][i]
                start, stop = data["Y"][i]

                spans = self.get_spans(context, context_tokens)
                idxs = []
                for word_idx, span in enumerate(spans):
                    if not (stop <= span[0] or start >= span[1]):
                        idxs.append(word_idx)
                
                data["Y"][i] = (idxs[0], idxs[-1] + 1)
        __transform_labels(self.train_data)
        __transform_labels(self.test_data)

    def get_spans(self, text, all_tokens):
        spans = []
        cur_idx = 0
        for token in all_tokens:
            if text.find(token, cur_idx) < 0:
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        return spans



def get_normal_format(dataset, cut_off=None):
    """
    reformat the dataset to normal version.
    """
    reformatted_data = []
    assert len(dataset["context_X"]) == len(dataset["question_X"]) == len(dataset["Y"]) == len(dataset["question_ids"])
    for c, q, a, qid in zip(dataset["context_X"], dataset["question_X"], dataset["Y"], dataset["question_ids"]):
        item = {}
        item["context"] = c
        item["qas"] = [
            {
                # "id": "%d"%(len(reformatted_data)+1),
                "id": qid,
                "is_impossible": False,
                "question": q,
                "answers": [{"text": c[a[0]:a[1]], "answer_start": a[0]}],
            }
        ]
        reformatted_data.append(item)
    return reformatted_data[:cut_off]