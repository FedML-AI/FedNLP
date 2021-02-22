from abc import ABC, abstractmethod


class BaseRawDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path):
        self.data_path = data_path
        self.index_list = None
        self.attributes = dict()

    @abstractmethod
    def data_loader(self):
        pass

    @abstractmethod
    def process_data(self, file_path):
        pass

    @abstractmethod
    def generate_h5_file(self, file_path):
        pass

class TextClassificationRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super.__init__(data_path)
        self.X = list()
        self.Y = list()
        self.num_labels = -1
        self.label_vocab = dict()
        self.task_type = "text_classification"
    
    def generate_h5_file(self, file_path):
        pass

class SpanExtractionRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super.__init__(data_path)
        self.document_X = list()
        self.question_X = list()
        self.Y = list()
        self.task_type = "span_extraction"

    def generate_h5_file(self, file_path):
        pass

class SeqTaggingRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super.__init__(data_path)
        self.X = list()
        self.Y = list()
        self.num_labels = -1
        self.label_vocab = dict()
        self.task_type = "seq_tagging"

    def generate_h5_file(self, file_path):
        pass

class Seq2SeqRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super.__init__(data_path)
        self.X = list()
        self.Y = list()
        self.task_type = "seq2seq"
    
    def generate_h5_file(self, file_path):
        pass



