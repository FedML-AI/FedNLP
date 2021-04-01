import pickle


from data_preprocessing.base.base_raw_data_loader import Seq2SeqRawDataLoader
from data_preprocessing.base.partition import *


class RawDataLoader(Seq2SeqRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0:
            total_size = self.process_data_file(self.data_path)
            self.attributes["index_list"] = [i for i in range(total_size)]

    def process_data_file(self, file_path):
        source_file_path = file_path[0]
        target_file_path = file_path[1]
        source_size = 0
        with open(source_file_path, "r") as f:
            for line in f:
                line = line.strip()
                idx = len(self.X)
                self.X[idx] = line
                source_size += 1
        target_size = 0
        with open(target_file_path, "r") as f:
            for line in f:
                line = line.strip()
                idx = len(self.Y)
                self.Y[idx] = line
                target_size += 1
        assert source_size == target_size
        return source_size


class ClientDataLoader(BaseClientDataLoader):
    def __init__(self, data_path, partition_path, language_pair, client_idx=None, partition_method="uniform",
                 tokenize=False):
        data_fields = ["X", "Y"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        self.language_pair = language_pair
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        source_tokenizer = self.spacy_tokenizer[self.language_pair[0] + "_tokenizer"]
        target_tokenizer = self.spacy_tokenizer[self.language_pair[1] + "_tokenizer"]

        def __tokenize_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = [token.text.strip().lower() for token in source_tokenizer(data["X"][i].strip()) if token.text.strip()]
                data["Y"][i] = [token.text.strip().lower() for token in target_tokenizer(data["Y"][i].strip()) if token.text.strip()]

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)


# if __name__ == "__main__":
#     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs",
#                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en"]
#     data_loader = RawDataLoader(data_file_paths)
#     results = data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#     pickle.dump(results, open("wmt_cs_en_data_loader.pkl", "wb"))
#     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_cs_en_partition.pkl", "wb"))

#     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.de",
#                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.en"]
#     data_loader = RawDataLoader(data_file_paths)
#     results = data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#     pickle.dump(results, open("wmt_de_en_data_loader.pkl", "wb"))
#     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_de_en_partition.pkl", "wb"))

#     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.ru",
#                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.en"]
#     data_loader = RawDataLoader(data_file_paths)
#     results = data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#     pickle.dump(results, open("wmt_ru_en_data_loader.pkl", "wb"))
#     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_ru_en_partition.pkl", "wb"))

#     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.zh",
#                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.en"]
#     data_loader = RawDataLoader(data_file_paths)
#     results = data_loader.data_loader()
#     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
#     pickle.dump(results, open("wmt_zh_en_data_loader.pkl", "wb"))
#     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_zh_en_partition.pkl", "wb"))
#     print("done")
