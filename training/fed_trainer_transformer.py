import logging

import torch
import pandas as pd
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from model.fed_transformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset
)
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset
import warnings
import os
import sklearn
from data_preprocessing.base.base_data_loader import BaseDataLoader
# import data_preprocessing.SQuAD_1_1.data_loader

class FedTransformerTrainer(ModelTrainer):

    def __init__(self, client_trainer, client_model, task_formulation="classification"):
        self.client_trainer = client_trainer
        self.client_model = client_model
        self.id = 0
        
    def get_model_params(self):
        return self.client_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.client_model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Client(%d)"%self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.client_trainer.train_dl = train_data
        self.client_trainer.train_model(device=device)


    def test(self, test_data, device, args=None):
        logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data)))
        self.client_trainer.test_dl = test_data
        self.client_trainer.eval_model(device=device)

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        global_test_data = []
        global_test_examples = []
        global_test_features = []
        for idx, local_test_dl in test_data_local_dict.items():
            local_data = local_test_dl.dataset
            local_examples = local_test_dl.examples
            local_features = local_test_dl.features
            global_test_data += local_data
            global_test_examples += local_examples
            global_test_features += local_features
            logging.info("client idx(%d) len data(%d) len examples(%d) len features(%d)" % (idx, len(local_data), len(local_examples), len(local_features)))
        global_test_dl = BaseDataLoader(global_test_examples, global_test_features, global_test_data,
                                batch_size=local_test_dl.batch_size,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
        # global_test_dl.to(device)

        logging.info("Client(%d)"%self.id + ":| Global Test Data Size = %d" % (len(global_test_data)))
        logging.info(len(global_test_examples))
        logging.info(len(global_test_features))
        self.client_trainer.test_dl = global_test_dl
        self.client_trainer.eval_model(device=device)
        return True
