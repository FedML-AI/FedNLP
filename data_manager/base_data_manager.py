from abc import ABC, abstractmethod
import h5py
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import h5py
import json
import numpy as np


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

        self.num_clients = self.load_num_clients(
            self.args.partition_file_path, self.args.partition_method)
        # TODO: sync to the same logic to sample index
        # self.client_index_list = self.sample_client_index(process_id, num_workers)
        # self.client_index_list = self.get_all_clients()
        self.client_index_list = self.sample_client_index(process_id, num_workers)

    @staticmethod
    def load_attributes(data_path):
        data_file = h5py.File(data_path, "r", swmr=True)
        attributes = json.loads(data_file["attributes"][()])
        data_file.close()
        return attributes

    @staticmethod
    def load_num_clients(partition_file_path, partition_name):
        data_file = h5py.File(partition_file_path, "r", swmr=True)
        num_clients = int(data_file[partition_name]["n_clients"][()])
        data_file.close()
        return num_clients

    @abstractmethod
    def read_instance_from_h5(self, data_file, index_list):
        pass

    def sample_client_index(self, process_id, num_workers):
        '''
        Sample client indices according to process_id
        '''
        # process_id = 0 means this process is the server process
        if process_id == 0:
            return None
        else:
            return self._simulated_sampling(process_id)

    def _simulated_sampling(self, process_id):
        res_client_indexes = list()
        for round_idx in range(self.args.comm_round):
            if self.num_clients == self.num_workers:
                client_indexes = [client_index
                                  for client_index in range(self.num_clients)]
            else:
                nc = min(self.num_workers, self.num_clients)
                # make sure for each comparison, we are selecting the same clients each round
                np.random.seed(round_idx)
                client_indexes = np.random.choice(
                    range(self.num_clients),
                    nc, replace=False)
                # logging.info("client_indexes = %s" % str(client_indexes))
            res_client_indexes.append(client_indexes[process_id-1])
        return res_client_indexes

    def get_all_clients(self):
        return list(range(0, self.num_clients))

    def load_centralized_data(self):
        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(
            self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method
        train_index_list = []
        test_index_list = []
        for client_idx in tqdm(
            partition_file[partition_method]
            ["partition_data"].keys(),
                desc="Loading data from h5 file."):
            train_index_list.extend(
                partition_file[partition_method]["partition_data"]
                [client_idx]["train"][()])
            test_index_list.extend(
                partition_file[partition_method]["partition_data"]
                [client_idx]["test"][()])
        train_X = [
            data_file["X"][str(idx)][()].decode("utf-8")
            for idx in train_index_list]
        train_y = [
            data_file["Y"][str(idx)][()].decode("utf-8")
            for idx in train_index_list]
        test_X = [
            data_file["X"][str(idx)][()].decode("utf-8")
            for idx in test_index_list]
        test_y = [
            data_file["Y"][str(idx)][()].decode("utf-8")
            for idx in test_index_list]
        data_file.close()
        partition_file.close()
        train_examples, train_dataset = self.preprocessor.transform(
            train_X, train_y, train_index_list)
        test_examples, test_dataset = self.preprocessor.transform(
            test_X, test_y, test_index_list, evaluate=True)
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
        partition_file = h5py.File(
            self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method
        train_index_list = []
        test_index_list = []
        test_data_local_dict = {}
        test_X = []
        test_y = []
        for client_idx in tqdm(
            partition_file[partition_method]
            ["partition_data"].keys(),
                desc="Loading data from h5 file."):
            train_index_list.extend(
                partition_file[partition_method]["partition_data"]
                [client_idx]["train"][()])
            local_test_index_list = partition_file[partition_method][
                "partition_data"][client_idx]["test"][()]
            test_index_list.extend(local_test_index_list)
            # local_test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in local_test_index_list]
            # local_test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in local_test_index_list]
            local_test_X, local_test_y = self.read_instance_from_h5(
                data_file, local_test_index_list)
            _, local_test_dataset = self.preprocessor.transform(
                local_test_X, local_test_y, local_test_index_list, evaluate=True)
            local_test_data = DataLoader(local_test_dataset,
                                         batch_size=self.eval_batch_size,
                                         num_workers=0,
                                         pin_memory=True,
                                         drop_last=False)
            test_data_local_dict[int(client_idx)] = local_test_data
            test_X += local_test_X
            test_y += local_test_y

        # train_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in train_index_list]
        # train_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list]
        train_X, train_y = self.read_instance_from_h5(
            data_file, train_index_list)
        # test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in test_index_list]
        # test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list]
        data_file.close()
        partition_file.close()

        train_examples, train_dataset = self.preprocessor.transform(
            train_X, train_y, train_index_list)
        test_examples, test_dataset = self.preprocessor.transform(
            test_X, test_y, test_index_list, evaluate=True)
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

        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)

    def _load_federated_data_local(self):

        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(
            self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method

        train_data_local_dict = {}
        test_data_local_dict = {}
        train_data_local_num_dict = {}
        self.client_index_list = list(set(self.client_index_list))
        logging.info("self.client_index_list = " + str(self.client_index_list))

        for client_idx in self.client_index_list:
            # TODO: cancel the partiation file usage
            train_index_list = partition_file[partition_method][
                "partition_data"][
                str(client_idx)]["train"][
                ()]
            test_index_list = partition_file[partition_method][
                "partition_data"][
                str(client_idx)]["test"][
                ()]
            train_X, train_y = self.read_instance_from_h5(
                data_file, train_index_list)
            test_X, test_y = self.read_instance_from_h5(
                data_file, test_index_list)
            # test_X = [data_file["X"][str(idx)][()].decode("utf-8") for idx in test_index_list]
            # test_y = [data_file["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list]
            train_examples, train_dataset = self.preprocessor.transform(
                train_X, train_y, train_index_list)
            test_examples, test_dataset = self.preprocessor.transform(
                test_X, test_y, test_index_list, evaluate=True)

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
        return (train_data_num, train_data_global, test_data_global,
                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.num_clients)
