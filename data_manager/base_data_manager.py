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
        self.client_index_list = None
        self.client_index_pointer = 0
        self.attributes = None

        self.client_index_list = self.sample_client_index(process_id, num_workers)

    def load_all_data(self):
        return self.__load_data()

    def load_client_data(self, client_idx):
        return self.__load_data(client_idx)

    @abstractmethod
    def __load_data(self, client_idx=None):
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
        '''
        load client data for next round training
        '''
        # if client_index_list is None and train_dataset is None, it means we need to load all data from scratch
        if self.client_index_list is None:
            if self.train_dataset is None:
                self.train_examples, self.test_examples, self.train_dataset, self.test_dataset = self.__load_data(None)
            return
        # client_index_pointer will point the client index, we use that index to load the client data
        self.train_examples, self.test_examples, self.train_dataset, self.test_dataset = self.__load_data(self.client_index_list[self.client_index_pointer])
        # move to the next client index
        self.client_index_pointer = self.client_index_pointer + 1 if self.client_index_pointer + 1 < len(self.client_index_list) else 0

    def sample_client_index(self, process_id, num_workers):
        '''
        Sample client indices according to process_id
        '''
        # process_id = 0 means this process is the server process
        if process_id == 0:
            return None
        else:
            num_clients = self.args.num_clients
            # get the number of clients per workers
            size = num_clients // num_workers
            start = (process_id - 1) * size
            end = process_id * size if num_workers != process_id else num_clients
            return [i for i in range(start, end)]