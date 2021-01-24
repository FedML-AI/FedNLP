import argparse
import json
import logging
import os
import random
import socket
import sys

import psutil
import setproctitle
import torch
import torch.nn
import wandb
from spacy.lang.en import STOP_WORDS

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from FedML.fedml_api.distributed.fedopt import FedOptAPI
from FedML.fedml_api.distributed.fedavg import FedAvgAPI
from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

import data_preprocessing.AGNews.data_loader
import data_preprocessing.SST_2.data_loader
import data_preprocessing.SemEval2010Task8.data_loader
import data_preprocessing.Sentiment140.data_loader
import data_preprocessing.news_20.data_loader
from data_preprocessing.base.utils import *
from model.bilstm import BiLSTM_TextClassification
from training.text_classification_bilstm_trainer import TextClassificationBiLSTMTrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='bilstm', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='sentiment140', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/sentiment_140_data_loader.pkl',
                        metavar="DF", help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='data/partition/sentiment_140_partition.pkl',
                        metavar="PF", help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--hidden_size', type=int, default=300, metavar='H',
                        help='size of hidden layers')

    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of layers in neural network')

    parser.add_argument('--lstm_dropout', type=float, default=0.1, metavar='LD',
                        help="dropout rate for LSTM's output")

    parser.add_argument('--embedding_dropout', type=float, default=0, metavar='ED',
                        help='dropout rate for word embedding')

    parser.add_argument('--attention_dropout', type=float, default=0, metavar='AD',
                        help='dropout rate for attention layer output, only work when BiLSTM_Attention is chosen')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='MSL',
                        help='maximum sequence length (-1 means the maximum sequence length in the dataset)')

    parser.add_argument('--embedding_file', type=str, default="data/pretrained/glove.840B.300d.txt",
                        metavar="EF", help='word embedding file')

    parser.add_argument('--embedding_name', type=str, default="glove", metavar="EN",
                        help='word embedding name(word2vec, glove)')

    parser.add_argument('--embedding_length', type=int, default=300, metavar="EL", help='dimension of word embedding')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--server_lr', type=float, default=0.001,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--fed_alg', type=str, default="fedavg", help="The training algorithm in federated learning")

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=1,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument("--do_remove_stop_words", type=lambda x: (str(x).lower() == 'true'), default=False,
                        metavar="RSW",
                        help="remove stop words which specify in sapcy")

    parser.add_argument('--do_remove_low_freq_words', type=int, default=5, metavar="RLW",
                        help='remove words in lower frequency')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    client_data_loaders = []
    if dataset_name == "20news":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.news_20.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.news_20.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_data_loaders.append(client_data_loader)
    elif dataset_name == "agnews":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.AGNews.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.AGNews.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_data_loaders.append(client_data_loader)
    elif dataset_name == "semeval_2010_task8":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.SemEval2010Task8.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.SemEval2010Task8.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_data_loaders.append(client_data_loader)
    elif dataset_name == "sentiment140":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.Sentiment140.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.Sentiment140.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_data_loaders.append(client_data_loader)
    elif dataset_name == "sst_2":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.SST_2.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.SST_2.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_data_loaders.append(client_data_loader)
    else:
        raise Exception("No such dataset")
    attributes = server_data_loader.get_attributes()
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    for idx in range(attributes["n_clients"]):
        train_data_local_num_dict[idx] = client_data_loaders[idx].get_train_data_num()
        train_data_local_dict[idx] = client_data_loaders[idx].get_train_batch_data(args.batch_size)
        test_data_local_dict[idx] = client_data_loaders[idx].get_test_batch_data(args.batch_size)
    dataset = [server_data_loader.get_train_data_num(), server_data_loader.get_test_data_num(),
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, attributes]
    return dataset


def load_distributed_data(process_id, args):
    """
    load distributed data from {dataset}_distributed generated by distributed_data_utils.py
    args: 
        process_id -- also referred as worker_id, indicates the index of worker in each training round
        args -- program arguments including information like dataset, comm_round etc
    """
    dataset_name = args.dataset
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    client_list = None

    info_file = 'data/distributed/{}_distributed/info.json'.format(args.dataset)
    with open(info_file, 'r') as f:
        info = json.load(f)
        assert info['comm_round'] == args.comm_round
        assert info['client_num_per_round'] == args.client_num_per_round
        assert info['data_file'] == args.data_file
        assert info['partition_file'] == args.partition_file

    worker_file = 'data/distributed/{}_distributed/{}.json'.format(args.dataset, process_id)
    with open(worker_file, 'r') as f:
        client_list = list(set(json.load(f)))
        if process_id - 1 not in client_list:
            client_list = [process_id - 1] + client_list # include client of (process_id - 1) to bootstrap trainer init
    
    logging.info("load_data. dataset_name = %s" % dataset_name)
    client_data_loaders = []
    if dataset_name == "20news":
        for client_index in client_list:
            data_loader = data_preprocessing.news_20.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            train_data_local_num_dict[client_index] = data_loader.get_train_data_num()
            train_data_local_dict[client_index] = data_loader.get_train_batch_data(args.batch_size)
            test_data_local_dict[client_index] = data_loader.get_test_batch_data(args.batch_size)
    elif dataset_name == "agnews":
        for client_index in client_list: 
            data_loader = data_preprocessing.AGNews.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            train_data_local_num_dict[client_index] = data_loader.get_train_data_num()
            train_data_local_dict[client_index] = data_loader.get_train_batch_data(args.batch_size)
            test_data_local_dict[client_index] = data_loader.get_test_batch_data(args.batch_size)
    elif dataset_name == "semeval_2010_task8":
        for client_index in client_list: 
            data_loader = data_preprocessing.SemEval2010Task8.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            train_data_local_num_dict[client_index] = data_loader.get_train_data_num()
            train_data_local_dict[client_index] = data_loader.get_train_batch_data(args.batch_size)
            test_data_local_dict[client_index] = data_loader.get_test_batch_data(args.batch_size)
    elif dataset_name == "sentiment140":
        for client_index in client_list: 
            data_loader = data_preprocessing.Sentiment140.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            train_data_local_num_dict[client_index] = data_loader.get_train_data_num()
            train_data_local_dict[client_index] = data_loader.get_train_batch_data(args.batch_size)
            test_data_local_dict[client_index] = data_loader.get_test_batch_data(args.batch_size)
    elif dataset_name == "sst_2":
        for client_index in client_list: 
            data_loader = data_preprocessing.SST_2.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            train_data_local_num_dict[client_index] = data_loader.get_train_data_num()
            train_data_local_dict[client_index] = data_loader.get_train_batch_data(args.batch_size)
            test_data_local_dict[client_index] = data_loader.get_test_batch_data(args.batch_size)
    else:
        raise Exception("No such dataset")
    attributes = data_loader.get_attributes()
    dataset = [data_loader.get_train_data_num(), data_loader.get_test_data_num(),
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, attributes]
    return dataset


def preprocess_data(args, dataset):
    """
    preprocessing data for further training, which includes load pretrianed embeddings, padding data and transforming
    token and label to index
    """
    logging.info("preproccess data")
    [train_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict,
     test_data_local_dict, attributes] = dataset

    target_vocab = attributes["target_vocab"]

    # remove low frequency words and stop words
    # build frequency vocabulary based on tokenized data
    x = []
    for client_index in train_data_local_num_dict.keys():
        for i, batch_data in enumerate(train_data_local_dict[client_index]):
            x.extend(batch_data["X"])
        for i, batch_data in enumerate(test_data_local_dict[client_index]):
            x.extend(batch_data["X"])
    freq_vocab = build_freq_vocab(x)
    logging.info("frequency vocab size %d", len(freq_vocab))

    def __remove_words(word_set):

        for client_index in train_data_local_num_dict.keys():
            for i, batch_data in enumerate(train_data_local_dict[client_index]):
                train_data_local_dict[client_index][i]["X"] = remove_words(batch_data["X"], word_set)

            for i, batch_data in enumerate(test_data_local_dict[client_index]):
                test_data_local_dict[client_index][i]["X"] = remove_words(batch_data["X"], word_set)

    if args.do_remove_low_freq_words > 0:
        logging.info("remove low frequency words")
        # build low frequency words set
        low_freq_words = set()
        for token, freq in freq_vocab.items():
            if freq <= args.do_remove_low_freq_words:
                low_freq_words.add(token)

        __remove_words(low_freq_words)

    if args.do_remove_stop_words:
        logging.info("remove stop words")
        __remove_words(STOP_WORDS)

    x.clear()
    for client_index in train_data_local_num_dict.keys():
        for i, batch_data in enumerate(train_data_local_dict[client_index]):
            x.extend(batch_data["X"])
        for i, batch_data in enumerate(test_data_local_dict[client_index]):
            x.extend(batch_data["X"])
    source_vocab = build_vocab(x)
    logging.info("source vocab size %d", len(source_vocab))

    # load pretrained embeddings. Note that we use source vocabulary here to reduce the input size
    embedding_weights = None
    if args.embedding_name:
        if args.embedding_name == "word2vec":
            logging.info("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_word2vec_embedding(os.path.abspath(args.embedding_file),
                                                                      source_vocab)
        elif args.embedding_name == "glove":
            logging.info("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_glove_embedding(os.path.abspath(args.embedding_file), source_vocab,
                                                                   args.embedding_length)
        else:
            raise Exception("No such embedding")
        embedding_weights = torch.tensor(embedding_weights, dtype=torch.float)

    if args.max_seq_len == -1:
        lengths = []
        for batch_data in train_data_global:
            lengths.extend([len(single_x) for single_x in batch_data["X"]])
        args.max_seq_len = max(lengths)

    new_train_data_local_dict = dict()
    new_test_data_local_dict = dict()

    # padding data and transforming token as well as label to index
    for client_index in train_data_local_num_dict.keys():
        new_train_data_local = list()
        for i, batch_data in enumerate(train_data_local_dict[client_index]):
            padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
            new_train_data_local.append(
                {"X": token_to_idx(padding_x, source_vocab),
                 "Y": label_to_idx(batch_data["Y"], target_vocab),
                 "seq_lens": seq_lens})
        new_train_data_local_dict[client_index] = new_train_data_local

        new_test_data_local = list()
        for i, batch_data in enumerate(test_data_local_dict[client_index]):
            padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
            new_test_data_local.append(
                {"X": token_to_idx(padding_x, source_vocab),
                 "Y": label_to_idx(batch_data["Y"], target_vocab),
                 "seq_lens": seq_lens})
        new_test_data_local_dict[client_index] = new_test_data_local

    train_global_data = []
    test_global_data = []
    for key in train_data_local_num_dict.keys():
        train_global_data.extend(train_data_local_dict[key])
        test_global_data.extend(test_data_local_dict[key])

    logging.info("size of source vocab: %s, size of target vocab: %s" % (len(source_vocab), len(target_vocab)))
    dataset = [train_data_num, test_data_num, train_global_data, test_global_data, train_data_local_num_dict,
               new_train_data_local_dict, new_test_data_local_dict, source_vocab, target_vocab, embedding_weights]
    return dataset


def preprocess_distributed_data(args, dataset):
    """
    distributed version of preprocess_data, loads source_vocab instead of retrieving all tokens from train/test_gloal_data
    """
    logging.info("preproccess data")
    [train_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict,
     test_data_local_dict, attributes] = dataset

    target_vocab = attributes["target_vocab"]
    source_vocab = dict()
    vocab_file = 'data/distributed/{}_distributed/vocab.json'.format(args.dataset)
    with open(vocab_file, 'r') as f:
        source_vocab = json.load(f)
        logging.info("source vocab size %d", len(source_vocab))

    # load pretrained embeddings. Note that we use source vocabulary here to reduce the input size
    embedding_weights = None
    if args.embedding_name:
        if args.embedding_name == "word2vec":
            logging.info("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_word2vec_embedding(os.path.abspath(args.embedding_file),
                                                                      source_vocab)
        elif args.embedding_name == "glove":
            logging.info("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_glove_embedding(os.path.abspath(args.embedding_file), source_vocab,
                                                                   args.embedding_length)
        else:
            raise Exception("No such embedding")
        embedding_weights = torch.tensor(embedding_weights, dtype=torch.float)

    if args.max_seq_len == -1:
        raise Exception("Max sequence length must be specified!")

    new_train_data_local_dict = dict()
    new_test_data_local_dict = dict()

    # padding data and transforming token as well as label to index
    for client_index in train_data_local_num_dict.keys():
        new_train_data_local = list()
        for i, batch_data in enumerate(train_data_local_dict[client_index]):
            padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
            new_train_data_local.append(
                {"X": token_to_idx(padding_x, source_vocab),
                 "Y": label_to_idx(batch_data["Y"], target_vocab),
                 "seq_lens": seq_lens})
        new_train_data_local_dict[client_index] = new_train_data_local

        new_test_data_local = list()
        for i, batch_data in enumerate(test_data_local_dict[client_index]):
            padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
            new_test_data_local.append(
                {"X": token_to_idx(padding_x, source_vocab),
                 "Y": label_to_idx(batch_data["Y"], target_vocab),
                 "seq_lens": seq_lens})
        new_test_data_local_dict[client_index] = new_test_data_local

    train_global_data = []
    test_global_data = []
    for key in train_data_local_num_dict.keys():
        train_global_data.extend(new_train_data_local_dict[key])
        test_global_data.extend(new_test_data_local_dict[key])
    logging.info("size of source vocab: %s, size of target vocab: %s" % (len(source_vocab), len(target_vocab)))
    dataset = [train_data_num, test_data_num, train_global_data, test_global_data, train_data_local_num_dict,
               new_train_data_local_dict, new_test_data_local_dict, source_vocab, target_vocab, embedding_weights]
    return dataset


def create_model(args, model_name, input_size, output_size, embedding_weights):
    logging.info("create_model. model_name = %s, input_size = %s, output_size = %s"
                 % (model_name, input_size, output_size))
    if model_name == "bilstm_attention":
        model = BiLSTM_TextClassification(input_size, args.hidden_size, output_size, args.num_layers,
                                          args.embedding_dropout, args.lstm_dropout, args.attention_dropout,
                                          args.embedding_length, attention=True, embedding_weights=embedding_weights)
    elif model_name == "bilstm":
        model = BiLSTM_TextClassification(input_size, args.hidden_size, output_size, args.num_layers,
                                          args.embedding_dropout, args.lstm_dropout, args.attention_dropout,
                                          args.embedding_length, embedding_weights=embedding_weights)
    else:
        raise Exception("No such model")
    return model


if __name__ == "__main__":
    logging.info("start")

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # initialize distributed computing (MPI)
    if args.fed_alg == "fedavg":
        comm, process_id, worker_number = FedAvgAPI.FedML_init()
    elif args.fed_alg == "fedopt":
        comm, process_id, worker_number = FedOptAPI.FedML_init()
    else:
        raise Exception("No such federated algorithm")

    # customize the log format
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # customize the process name
    str_process_name = "FedNLP (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fednlp",
            name="FedOpt-" + str(args.partition_method) + "-r" + str(args.comm_round) + "-e" + str(
                args.epochs) + "-lr" + str(
                args.lr) + "-" + str(args.dataset),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # check "gpu_mapping.yaml" to see how to define the topology
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, \
                                                            args.gpu_mapping_file, args.gpu_mapping_key)
                                                            
    logging.info("process_id = %d, size = %d, device=%s" % (process_id, worker_number, str(device)))

    # load data
    preprocessed_sampling_lists = None
    if process_id == 0:
        dataset = load_data(args, args.dataset)
        worker_lists = []
        for i in range(args.client_num_per_round) :
            worker_idx = i + 1
            worker_file = 'data/distributed/{}_distributed/{}.json'.format(args.dataset, worker_idx)
            with open(worker_file, 'r') as f:
                client_list = json.load(f)
                worker_lists.append(client_list)
        preprocessed_sampling_lists = list(map(list, zip(*worker_lists))) # return [[round1 clients list], [round2 clients list], ...]
    else:
        dataset = load_distributed_data(process_id, args)
    
    dataset = preprocess_distributed_data(args, dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict,
     train_data_local_dict,
     test_data_local_dict, source_vocab, target_vocab, embedding_weights] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, input_size=len(source_vocab), output_size=len(target_vocab),
                         embedding_weights=embedding_weights)

    # define the customized model trainer using abstract class `FedML.fedml_core.trainer.model_trainer`
    model_trainer = TextClassificationBiLSTMTrainer(model)

    # start federated algorithm
    # for distributed algorithm, train_data_gloabl and test_data_global are required
    if args.fed_alg == "fedavg":
        FedAvgAPI.FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                             model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,
                             model_trainer, preprocessed_sampling_lists=preprocessed_sampling_lists)
    elif args.fed_alg == "fedopt":
        FedOptAPI.FedML_FedOpt_distributed(process_id, worker_number, device, comm,
                                           model, train_data_num, train_data_global, test_data_global,
                                           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,
                                           model_trainer, preprocessed_sampling_lists=preprocessed_sampling_lists)
    else:
        raise Exception("No such federated algorithm")
    logging.info("end")
