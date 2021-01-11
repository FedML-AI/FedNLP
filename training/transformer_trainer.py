import logging

import torch
import pandas as pd
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from model.fed_transformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import warnings
import os

class TransformerTrainer(ModelTrainer):

    def __init__(self, transformer_model):
        self.transformer_model = transformer_model
        self.model = self.transformer_model.model
        self.id = 0
        
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def train(self, train_data, device, args):
        self.device = device
        self.transformer_model.device = device
        self.transformer_model._move_model_to_device() 
        labels_map = self.transformer_model.labels_map
        train_data_flat = dict(X=[], Y=[])

        for item in train_data: 
            train_data_flat["X"] += [t for t in item["X"]]
            train_data_flat["Y"] += [t for t in item["Y"]]
            break
        train_data = [(x, labels_map[y])
                  for x, y in zip(train_data_flat["X"], train_data_flat["Y"])]
        train_df = pd.DataFrame(train_data)
        global_step, training_details = self.transformer_model.train_model(train_df=train_df, client_desc="Client(%d)"%self.id)

        # after the first round, we should turn off the caching for speeding up
        self.transformer_model.args.reprocess_input_data = False


    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        # TODO:
        # return test_acc, test_total, test_loss
        # self.transformer_model.eval_model()
        return True
    
