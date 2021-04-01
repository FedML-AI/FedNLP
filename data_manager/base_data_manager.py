from abc import ABC, abstractmethod


class BaseDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, process_id, num_workers):
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.process_id = process_id
        self.num_workers = num_workers


        # TODO: add type comments for the below vars.
        self.train_dataset = None   
        self.test_dataset = None
        self.train_examples = None
        self.test_examples = None
        self.train_loader = None
        self.test_loader = None
        self.client_index = None
        self.client_index_pointer = 0
        self.attributes = None

        self.client_index = self.sample_client_index(process_id, num_workers)

    @abstractmethod
    def load_data(self, client_idx=None):
        pass

    @abstractmethod
    def load_attributes(self):
        pass

    def get_dataset(self):
        return self.train_examples, self.test_examples, self.train_dataset, self.test_dataset

    @abstractmethod
    def get_data_loader(self):
        pass

    def load_next_round_data(self):
        # TODO: add comments for the logic.
        if self.client_index is None:
            if self.train_dataset is None:
                self.train_examples, self.test_examples, self.train_dataset, self.test_dataset = self.load_data(self.client_index)
            return
        self.train_examples, self.test_examples, self.train_dataset, self.test_dataset = self.load_data(self.client_index[self.client_index_pointer])
        self.client_index_pointer = self.client_index_pointer + 1 if self.client_index_pointer + 1 < len(self.client_index) else 0

    def sample_client_index(self, process_id, num_workers):
        # TODO: add comments for the logic.
        if process_id == 0:
            return None
        else:
            num_clients = self.args.num_clients
            size = num_clients // num_workers
            start = (process_id - 1) * size
            end = process_id * size if num_workers != process_id else num_clients
            return [i for i in range(start, end)]