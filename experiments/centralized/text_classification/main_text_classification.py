import argparse
import logging
import random
import sys
import os

import numpy as np
import torch

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import fedml_api.data_preprocessing.fednlp.news_20.data_loader
import fedml_api.data_preprocessing.fednlp.AGNews.data_loader
import fedml_api.data_preprocessing.fednlp.SemEval2010Task8.data_loader
import fedml_api.data_preprocessing.fednlp.Sentiment140.data_loader
import fedml_api.data_preprocessing.fednlp.SST_2.data_loader

from fedml_api.model.nlp.rnn import BiLSTM
from fedml_api.data_preprocessing.fednlp.base.utils import *

from fedml_api.centralized.fednlp_classification.FedNLPClassificationAPI import FedNLP_Classification_centralized


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='bilstm_attention', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='20news', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='../../../data/fednlp/data_loaders/20news_data_loader.pkl',
                        help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='../../../data/fednlp/partition/20news_partition.pkl',
                        help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
                        help='how to partition the dataset')

    parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                        help='size of hidden layers')

    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of layers in neural network')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                        help='dropout rate for neural network')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--max_seq_len', type=int, default=-1, metavar='N',
                        help='maximum sequence length (-1 means the maximum sequence length in the dataset)')

    parser.add_argument('--embedding_file', type=str, default='', help='word embedding file')

    parser.add_argument('--embedding_name', type=str, default='', help='word embedding name')

    parser.add_argument('--embedding_length', type=int, default=300, help='dimension of word embedding')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--device', type=str, default=None, help='gpu device for training')

    args = parser.parse_args()

    return args


def load_data(args, dataset_name):
    data_loader = None
    if dataset_name == "20news":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        data_loader = fedml_api.data_preprocessing.fednlp.news_20.data_loader.\
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "agnews":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        data_loader = fedml_api.data_preprocessing.fednlp.AGNews.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "semeval_2010_task8":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        data_loader = fedml_api.data_preprocessing.fednlp.SemEval2010Task8.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "sentiment140":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        data_loader = fedml_api.data_preprocessing.fednlp.Sentiment140.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "sst_2":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        data_loader = fedml_api.data_preprocessing.fednlp.SST_2.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    else:
        raise Exception("No such dataset")
    dataset = [data_loader.get_train_batch_data(args.batch_size), data_loader.get_test_batch_data(args.batch_size),
               data_loader.get_attributes()]
    return dataset


def preprocess_data(args, dataset):
    source_vocab = None
    embedding_weights = None
    if args.embedding_file != '':
        source_vocab = dict()
        embedding_weights = []
    train_batch_data_list, test_batch_data_list, attributes = dataset
    target_vocab = attributes["target_vocab"]

    if source_vocab is None:
        x = []
        for batch_data in train_batch_data_list:
            x.extend(batch_data["X"])
        source_vocab = build_vocab(x)

    if args.max_seq_len == -1:
        lengths = []
        for batch_data in train_batch_data_list:
            lengths.extend([len(single_x) for single_x in batch_data["X"]])
        args.max_seq_len = max(lengths)
    new_train_batch_data_list = list()
    new_test_batch_data_list = list()
    for i, batch_data in enumerate(train_batch_data_list):
        new_train_batch_data_list.append({"X": token_to_idx(padding_data(batch_data["X"], args.max_seq_len), source_vocab),
                                          "Y": label_to_idx(batch_data["Y"], target_vocab)})

    for batch_data in test_batch_data_list:
        new_test_batch_data_list.append({"X": token_to_idx(padding_data(batch_data["X"], args.max_seq_len), source_vocab),
                                         "Y": label_to_idx(batch_data["Y"], target_vocab)})

    return new_train_batch_data_list, new_test_batch_data_list, source_vocab, target_vocab, embedding_weights


def create_model(args, model_name, input_size, output_size, embedding_weights):
    logging.info("create_model. model_name = %s, input_size = %s, output_size = %s"
                 % (model_name, input_size, output_size))
    model = None
    if model_name == "bilstm_attention":
        model = BiLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout, args.embedding_length,
                       attention=True, embedding_weights=embedding_weights)
    elif model_name == "bilstm":
        model = BiLSTM(input_size, args.hidden_size, output_size, args.num_layers, args.dropout, args.embedding_length,
                       embedding_weights=embedding_weights)
    else:
        raise Exception("No such model")
    return model


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)
    dataset = preprocess_data(args, dataset)

    # create model
    model = create_model(args, model_name=args.model, input_size=len(dataset[2]), output_size=len(dataset[3]),
                         embedding_weights=dataset[4])

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'

    FedNLP_Classification_centralized(model, dataset[0], dataset[1], args)

