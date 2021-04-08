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
import sklearn
import data_preprocessing.base.utils
# import data_preprocessing.SQuAD_1_1.data_loader

class FedTransformerTrainer(ModelTrainer):

    def __init__(self, client_trainer, client_model, task_formulation="classification"):
        self.client_trainer = client_trainer
        self.client_model = client_model
        self.id = 0
        assert task_formulation in ["classification", "sequence_tagging", "question_answering", "seq2seq"]
        self.task_formulation = task_formulation
        
    def get_model_params(self):
        return self.client_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.client_model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Client(%d)"%self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.client_trainer.train_dl = train_data
        self.client_trainer.train_model()



    def test(self, test_data, device, args=None):
        logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data)))
        self.client_trainer.test_dl = test_data
        self.client_trainer.eval_model()

        # if self.task_formulation == "classification":
        #     test_data_flat = self.flatten_classification_data(test_data)
        #     logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data_flat)))
        #     test_df = pd.DataFrame(test_data_flat)
        #     result, model_outputs, wrong_predictions = self.client_trainer.eval_model(test_df, acc=sklearn.metrics.accuracy_score) 
        #     logging.info("Client(%d)"%self.id + ":| Local Test Evaluation Result =%s" % (str(result)))
        # elif self.task_formulation == "sequence_tagging":
        #     test_data_flat = self.flatten_sequence_tagging_data(test_data)
        #     logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data_flat)))
        #     test_df = pd.DataFrame(test_data_flat, columns=["sentence_id", "words", "labels"])
        #     result, model_outputs, preds_list = self.client_trainer.eval_model(test_df) 
        #     logging.info("Client(%d)"%self.id + ":| Local Test Evaluation Result =%s" % (str(result)))
        # elif self.task_formulation == "question_answering":
        #     # TODO: 
        #     pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        global_test_data = []
        for idx, local_test_data in test_data_local_dict.items():
            global_test_data += local_test_data
        
        if self.task_formulation == "classification":
            test_data_flat = self.flatten_classification_data(global_test_data)
            logging.info("Client(%d)"%self.id + ":| Global Test Data Size = %d" % (len(test_data_flat)))
            test_df = pd.DataFrame(test_data_flat)
            result, model_outputs, wrong_predictions = self.client_trainer.eval_model(test_df, acc=sklearn.metrics.accuracy_score) 
            logging.info("Client(%d)"%self.id + ":| Global Test Evaluation Result =%s" % (str(result)))
        elif self.task_formulation == "sequence_tagging":
            test_data_flat = self.flatten_sequence_tagging_data(global_test_data)
            logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data_flat)))
            test_df = pd.DataFrame(test_data_flat, columns=["sentence_id", "words", "labels"])
            result, model_outputs, preds_list = self.client_trainer.eval_model(test_df) 
            logging.info("Client(%d)"%self.id + ":| Local Test Evaluation Result =%s" % (str(result)))
        elif self.task_formulation == "question_answering":
            # TODO: 
            pass
        return True
    
