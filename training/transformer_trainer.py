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
import data_preprocessing.SQuAD_1_1.data_loader

class TransformerTrainer(ModelTrainer):

    def __init__(self, transformer_model, task_formulation="classification"):
        self.transformer_model = transformer_model
        self.model = self.transformer_model.model
        self.id = 0
        assert task_formulation in ["classification", "sequence_tagging", "question_answering", "seq2seq"]
        self.task_formulation = task_formulation
        
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def flatten_classification_data(self, train_data):
        labels_map = self.transformer_model.labels_map
        train_data_flat = dict(X=[], Y=[])
        for item in train_data: 
            train_data_flat["X"] += [t for t in item["X"]]
            train_data_flat["Y"] += [t for t in item["Y"]]
        train_data_flat = [(x, labels_map[y])
                  for x, y in zip(train_data_flat["X"], train_data_flat["Y"])]
        return train_data_flat
    
    def flatten_question_answering_data(self, train_data):
        train_data_flat = dict(context_X=[], question_X=[], Y=[])
        for item in train_data: 
            train_data_flat["context_X"] += [t for t in item["context_X"]]
            train_data_flat["question_X"] += [t for t in item["question_X"]]
            train_data_flat["Y"] += [t for t in item["Y"]]
        return train_data_flat

    def flatten_sequence_tagging_data(self, train_data): 
        train_data_flat = dict(X=[], Y=[])
        for item in train_data: 
            train_data_flat["X"] += [t for t in item["X"]]
            train_data_flat["Y"] += [t for t in item["Y"]]
        train_data_flat = data_preprocessing.base.utils.NER_data_formatter(train_data_flat)
        return train_data_flat


    def train(self, train_data, device, args):
        self.device = device
        self.transformer_model.device = device
        self.transformer_model._move_model_to_device() 
        
        if self.task_formulation == "classification":
            train_data_flat = self.flatten_classification_data(train_data)
            logging.info("Client(%d)"%self.id + ":| Local Train Data Size = %d" % (len(train_data_flat)))
            train_df = pd.DataFrame(train_data_flat)
            global_step, training_details = self.transformer_model.train_model(train_df=train_df, client_desc="Client(%d)"%self.id)
        elif self.task_formulation == "sequence_tagging":
            train_data_flat = self.flatten_sequence_tagging_data(train_data) 
            logging.info("Client(%d)"%self.id + ":| Local Train Data Size = %d" % (len(train_data_flat)))
            train_df = pd.DataFrame(train_data_flat, columns=["sentence_id", "words", "labels"])
            global_step, training_details = self.transformer_model.train_model(train_data=train_df, client_desc="Client(%d)"%self.id)
        elif self.task_formulation == "question_answering":
            train_data_flat = self.flatten_question_answering_data(train_data)
            train_data_flat = data_preprocessing.SQuAD_1_1.data_loader.get_normal_format(train_data_flat)
            logging.info("Client(%d)"%self.id + ":| Local Train Data Size = %d" % (len(train_data_flat)))
            train_df = pd.DataFrame(train_data_flat)
            global_step, training_details = self.transformer_model.train_model(train_data=train_data_flat, client_desc="Client(%d)"%self.id)
        
        # self.transformer_model.args.reprocess_input_data = False


    def test(self, test_data, device, args=None):
        if self.task_formulation == "classification":
            test_data_flat = self.flatten_classification_data(test_data)
            logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data_flat)))
            test_df = pd.DataFrame(test_data_flat)
            result, model_outputs, wrong_predictions = self.transformer_model.eval_model(test_df, acc=sklearn.metrics.accuracy_score) 
            logging.info("Client(%d)"%self.id + ":| Local Test Evaluation Result =%s" % (str(result)))
        elif self.task_formulation == "sequence_tagging":
            test_data_flat = self.flatten_sequence_tagging_data(test_data)
            logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data_flat)))
            test_df = pd.DataFrame(test_data_flat, columns=["sentence_id", "words", "labels"])
            result, model_outputs, preds_list = self.transformer_model.eval_model(test_df) 
            logging.info("Client(%d)"%self.id + ":| Local Test Evaluation Result =%s" % (str(result)))
        elif self.task_formulation == "question_answering":
            # TODO: 
            pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        global_test_data = []
        for idx, local_test_data in test_data_local_dict.items():
            global_test_data += local_test_data
        
        if self.task_formulation == "classification":
            test_data_flat = self.flatten_classification_data(global_test_data)
            logging.info("Client(%d)"%self.id + ":| Global Test Data Size = %d" % (len(test_data_flat)))
            test_df = pd.DataFrame(test_data_flat)
            result, model_outputs, wrong_predictions = self.transformer_model.eval_model(test_df, acc=sklearn.metrics.accuracy_score) 
            logging.info("Client(%d)"%self.id + ":| Global Test Evaluation Result =%s" % (str(result)))
        elif self.task_formulation == "sequence_tagging":
            test_data_flat = self.flatten_sequence_tagging_data(global_test_data)
            logging.info("Client(%d)"%self.id + ":| Local Test Data Size = %d" % (len(test_data_flat)))
            test_df = pd.DataFrame(test_data_flat, columns=["sentence_id", "words", "labels"])
            result, model_outputs, preds_list = self.transformer_model.eval_model(test_df) 
            logging.info("Client(%d)"%self.id + ":| Local Test Evaluation Result =%s" % (str(result)))
        elif self.task_formulation == "question_answering":
            # TODO: 
            pass
        return True
    
