from FedML.fedml_api.data_preprocessing.MNIST.mnist_mobile_preprocessor import client_sampling
from data_manager.base_data_manager import BaseDataManager
from torch.utils.data import DataLoader
import h5py
import json
import logging
from tqdm import tqdm

class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""
    def __init__(self, args, model_args, process_id=0, num_workers=1, preprocessor=None):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(TextClassificationDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        # TODO: maybe add this back, currently as the external usage
        # self.load_next_round_data()
        # self.train_loader, self.test_loader = self.get_data_loader()

    def load_centralized_data(self):
        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method
        train_index_list = []
        test_index_list = []
        for client_idx in tqdm(partition_file[partition_method]["partition_data"].keys(), desc="Loading data from h5 file."):
            train_index_list.extend(partition_file[partition_method]["partition_data"][client_idx]["train"][()])
            test_index_list.extend(partition_file[partition_method]["partition_data"][client_idx]["test"][()])
        train_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in train_index_list]
        train_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list]
        test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in test_index_list]
        test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list]
        data_file.close()
        partition_file.close()
        train_examples, train_dataset = self.preprocessor.transform(train_X, train_y, train_index_list)
        test_examples, test_dataset = self.preprocessor.transform(test_X, test_y, test_index_list, evaluate=True)
        train_dl = DataLoader(train_dataset,
                                       batch_size=self.train_batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=False)

        test_dl = DataLoader(test_dataset,
                                    batch_size=self.eval_batch_size,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False)
        return train_examples, test_examples, train_dl, test_dl


    def load_federated_data(self, process_id):
        if process_id == 0:
            return self._load_federated_data_server()
        else:
            return self._load_federated_data_local()

    def _load_federated_data_server(self):
        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method
        train_index_list = []
        test_index_list = []
        test_data_local_dict = {}
        test_X = []
        test_y = []
        for client_idx in tqdm(partition_file[partition_method]["partition_data"].keys(), desc="Loading data from h5 file."):
            train_index_list.extend(partition_file[partition_method]["partition_data"][client_idx]["train"][()])
            local_test_index_list = partition_file[partition_method]["partition_data"][client_idx]["test"][()]
            test_index_list.extend(local_test_index_list)
            local_test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in local_test_index_list]
            local_test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in local_test_index_list]
            local_test_dataset = self.preprocessor.transform(local_test_X, local_test_y, local_test_index_list, evaluate=True)
            local_test_data = DataLoader(local_test_dataset,
                                    batch_size=self.eval_batch_size,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False)
            test_data_local_dict[int(client_idx)] = local_test_data
            test_X += local_test_X
            test_y += local_test_y

        train_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in train_index_list]
        train_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list]
        # test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in test_index_list]
        # test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list]
        data_file.close()
        partition_file.close()

        train_examples, train_dataset = self.preprocessor.transform(train_X, train_y, train_index_list)
        test_examples, test_dataset = self.preprocessor.transform(test_X, test_y, test_index_list, evaluate=True)
        train_data_global = DataLoader(train_dataset,
                                       batch_size=self.train_batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=False)

        test_data_global = DataLoader(test_dataset,
                                    batch_size=self.eval_batch_size,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False)
        train_data_num = len(train_examples)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        train_data_local_num_dict = None
        
        return (train_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)

    def _load_federated_data_local(self):
        
        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method

        train_data_local_dict = {}
        test_data_local_dict = {}
        train_data_local_num_dict = {}
        logging.info("self.client_index_list = " + str(self.client_index_list))
        
        for client_idx in self.client_index_list:
            # TODO: cancel the partiation file usage
            train_index_list = partition_file[partition_method]["partition_data"][str(client_idx)]["train"][()]
            test_index_list = partition_file[partition_method]["partition_data"][str(client_idx)]["test"][()]
            train_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in train_index_list]
            train_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list]
            test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in test_index_list]
            test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list]
            train_examples, train_dataset = self.preprocessor.transform(train_X, train_y, train_index_list)
            test_examples, test_dataset = self.preprocessor.transform(test_X, test_y, test_index_list, evaluate=True)

            train_loader = DataLoader(train_dataset,
                                       batch_size=self.train_batch_size,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=False)

            test_loader = DataLoader(test_dataset,
                                      batch_size=self.eval_batch_size,
                                      num_workers=0,
                                      pin_memory=True,
                                      drop_last=False)
            train_data_local_dict[client_idx] = train_loader
            test_data_local_dict[client_idx] = test_loader
            train_data_local_num_dict[client_idx] = len(train_examples)

        data_file.close()
        partition_file.close()
        
        train_data_global, test_data_global, train_data_num = None, None, 0
        return (train_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)

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



