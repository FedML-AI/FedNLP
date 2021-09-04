"""
Utils for generating distributed datasets across different workers 
Usage:
    python data_preprocessing/base/distributed_data_util.py  
    --dataset 20news 
    --data_file data/data_loaders/20news_data_loader.pkl 
    --partition_file data/partition/20news_partition.pkl 
    --partition_method uniform 
    --client_num_per_round 10 
    --comm_round 1000
"""
import argparse
import json
import logging
import os
import random
import sys

import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

import data_preprocessing.AGNews.data_loader
import data_preprocessing.SST_2.data_loader
import data_preprocessing.SemEval2010Task8.data_loader
import data_preprocessing.Sentiment140.data_loader
import data_preprocessing.news_20.data_loader
from data_preprocessing.base.utils import *


def add_args(parser):
    parser.add_argument('--dataset', type=str, default='sentiment140', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/sentiment_140_data_loader.pkl',
                        metavar="DF", help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='data/partition/sentiment_140_partition.pkl',
                        metavar="PF", help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we should use')

    args = parser.parse_args()
    return args


def load_data(args, idx=0):
    dataset_name = args.dataset
    print("load_data. dataset_name = %s" % dataset_name)
    if dataset_name == "20news":
        return data_preprocessing.news_20.data_loader. \
        ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                            partition_method=args.partition_method, tokenize=True, client_idx=idx)
    elif dataset_name == "agnews":
        return data_preprocessing.AGNews.data_loader. \
        ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                            partition_method=args.partition_method, tokenize=True, client_idx=idx)
    elif dataset_name == "semeval_2010_task8":
        return data_preprocessing.SemEval2010Task8.data_loader. \
        ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                            partition_method=args.partition_method, tokenize=True, client_idx=idx)
    elif dataset_name == "sentiment140":
        return data_preprocessing.Sentiment140.data_loader. \
        ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                            partition_method=args.partition_method, tokenize=True, client_idx=idx)
    elif dataset_name == "sst_2":
        return data_preprocessing.SST_2.data_loader. \
        ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                            partition_method=args.partition_method, tokenize=True, client_idx=idx)
    else:
        raise Exception("No such dataset")
    return data_preprocessing.Sentiment140.data_loader. \
        ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                            partition_method=args.partition_method, tokenize=True, client_idx=idx)


def client_sampling_all_rounds(comm_round, client_num_in_total, client_num_per_round):

    def client_sampling(round_idx):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            # may select fixed random seeds for comparison, e.g. random.seed(round_idx)
            client_indexes = random.sample(range(client_num_in_total), num_clients)
        print("client_indexes = %s" % str(client_indexes))
        return client_indexes
    
    sample_lists = [[] for i in range(comm_round)]
    for round_idx in range(comm_round):
        samples = client_sampling(round_idx)
        for worker_idx in range(len(samples)):
            sample_lists[round_idx] = samples

    return sample_lists


def generate_source_vocab(args, do_remove_low_freq_words=5, do_remove_stop_words=0):
    """
    preprocess global dataset to generate source vocab
    """
    print("generate source vocab...")
    # remove low frequency words and stop words
    # build frequency vocabulary based on tokenized data
    data_loader = load_data(args, None)
    x = []
    train_x = data_loader.get_train_batch_data()["X"]
    test_x = data_loader.get_test_batch_data()["X"]
    x.extend(train_x)
    x.extend(test_x)
    freq_vocab = build_freq_vocab(x)
    print("frequency vocab size %d", len(freq_vocab))

    if do_remove_low_freq_words > 0:
        print("remove low frequency words")
        # build low frequency words set
        low_freq_words = set()
        for token, freq in freq_vocab.items():
            if freq <= do_remove_low_freq_words:
                low_freq_words.add(token)
        train_x = remove_words(train_x, low_freq_words)
        test_x = remove_words(test_x, low_freq_words)

    if do_remove_stop_words:
        print("remove stop words")
        __remove_words(STOP_WORDS)

    x.clear()
    x.extend(train_x)
    x.extend(test_x)
    source_vocab = build_vocab(x)
    print("source vocab size %d", len(source_vocab))

    return source_vocab


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = add_args(parser)

    dataset = load_data(args)
    attributes = dataset.get_attributes()

    sample_lists = client_sampling_all_rounds(args.comm_round, attributes["n_clients"], args.client_num_per_round)
    
    worker_list = list(map(list, zip(*sample_lists))) # transpose 
    worker_num = len(worker_list)

    os.makedirs('data/distributed/{}_distributed'.format(args.dataset), exist_ok=True)

    info_file = 'data/distributed/{}_distributed/info.json'.format(args.dataset)
    info = vars(args)
    with open(info_file, 'w') as f:
            json.dump(info, f)

    for worker_idx in tqdm.tqdm(range(worker_num)):
        worker_file = 'data/distributed/{}_distributed/{}.json'.format(args.dataset, worker_idx + 1)
        with open(worker_file, 'w') as f:
            json.dump(worker_list[worker_idx], f)

    # source vocab
    vocab_file = 'data/distributed/{}_distributed/vocab.json'.format(args.dataset)
    if os.path.exists(vocab_file):
        print("vocab_file %s exists, skip generating..."%vocab_file)
    else:
        source_vocab = generate_source_vocab(args)
        with open(vocab_file, 'w') as f:
            json.dump(source_vocab, f)