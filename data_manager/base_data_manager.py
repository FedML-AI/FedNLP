from abc import ABC, abstractmethod


class BaseDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, perprocessor, client_idx=None):
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.perprocessor = perprocessor
        self.client_idx = client_idx

        self.train_loader = None
        self.test_loader = None
        self.n_clients = -1

    @abstractmethod
    def load_data(self, data_file_path, partition_file_path, client_idx=None):
        pass

    def get_dataset(self):
        return self.train_dataset, self.test_dataset, self.test_examples

    @abstractmethod
    def get_data_loader(self):
        pass

    def reload_data_with_client_idx(self, client_idx):
        self.client_idx = client_idx
        self.train_dataset, self.test_dataset, self.test_examples = self.load_data(
            data_file_path=self.model_args.data_file_path, 
            partition_file_path=self.model_args.partition_file_path,
            client_idx=self.client_idx)
        self.train_loader = None
        self.test_loader = None