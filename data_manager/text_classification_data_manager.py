from data_manager.base_data_manager import BaseDataManager
from torch.utils.data import DataLoader
import h5py
import json
import logging


class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""
    def __init__(self, args, model_args, process_id, num_workers, preprocessor):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(TextClassificationDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        # TODO: maybe add this back, currently as the external usage
        # self.load_next_round_data()
        # self.train_loader, self.test_loader = self.get_data_loader()

    # TODO: seperate to 2 functions: load_all_data / load_client_data
    def load_data(self, client_idx=None):
        logging.info("start loading data")
        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)

        partition_method = self.args.partition_method

        if client_idx is None:
            # Load all data from the file.
            train_index_list = []
            test_index_list = []
            # TODO: can this be improved?
            for client_idx in partition_file[partition_method]["partition_data"].keys():
                # TODO: add a progress bar?
                train_index_list.extend(partition_file[partition_method]["partition_data"][client_idx]["train"][()])
                test_index_list.extend(partition_file[partition_method]["partition_data"][client_idx]["test"][()])
        else:
            # Load the data with the specfic client_index.
            train_index_list = partition_file[partition_method]["partition_data"][client_idx]["train"][()]
            test_index_list = partition_file[partition_method]["partition_data"][client_idx]["test"][()]
        
        train_X = [data_file["X"][str(idx)][()].decode("utf8") for idx in train_index_list]
        train_y = [data_file["Y"][str(idx)][()].decode("utf8") for idx in train_index_list]
        test_X = [data_file["X"][str(idx)][()].decode("utf8") for idx in test_index_list]
        test_y = [data_file["Y"][str(idx)][()].decode("utf8") for idx in test_index_list]

        data_file.close()
        partition_file.close()

        train_examples, train_dataset = self.preprocessor.transform(train_X, train_y, train_index_list)

        test_examples, test_dataset = self.preprocessor.transform(test_X, test_y, test_index_list, evaluate=True)

        return train_examples, test_examples, train_dataset, test_dataset
    
    def load_data_fedml(self, worker_id, num_workers):
        
        return (train_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict)


    @staticmethod
    def load_attributes(data_path):
        data_file = h5py.File(data_path, "r", swmr=True)
        attributes = json.loads(data_file["attributes"][()])
        data_file.close()
        return attributes

    def get_data_loader(self):
        if self.train_loader is not None:
            del self.train_loader

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.train_batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=False)

        # TEST
        if self.test_loader is not None:
            del self.test_loader
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.eval_batch_size,
                                      num_workers=0,
                                      pin_memory=True,
                                      drop_last=False)
        return self.train_loader, self.test_loader



