import argparse
import os
import random
import logging

import torch
import torch.nn.functional as F
import wandb
from torch.optim import *
from spacy.lang.en import STOP_WORDS
from statistics import *

import data_preprocessing.AGNews.data_loader
import data_preprocessing.SST_2.data_loader
import data_preprocessing.SemEval2010Task8.data_loader
import data_preprocessing.Sentiment140.data_loader
import data_preprocessing.news_20.data_loader
from data_preprocessing.base.utils import *
from model.bilstm import BiLSTM_TextClassification


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='bilstm', metavar='M',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='20news', metavar='DS',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/20news_data_loader.pkl',
                        metavar="DF", help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='data/partition/20news_partition.pkl',
                        metavar="PF", help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='PM',
                        help='how to partition the dataset')

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

    parser.add_argument('--batch_size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='MSL',
                        help='maximum sequence length (-1 means the maximum sequence length in the dataset)')

    parser.add_argument('--embedding_file', type=str, nargs="?", metavar="EF", help='word embedding file')

    parser.add_argument('--embedding_name', type=str, nargs="?", metavar="EN",
                        help='word embedding name(word2vec, glove)')

    parser.add_argument('--embedding_length', type=int, default=300, metavar="EL", help='dimension of word embedding')

    parser.add_argument('--optimizer', type=str, default='adam', metavar="O",
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--device', type=str, default="cuda:3", metavar="DV", help='gpu device for training')

    parser.add_argument("--do_remove_stop_words", type=lambda x: (str(x).lower() == 'true'), default=False, metavar="RSW",
                        help="remove stop words which specify in sapcy")

    parser.add_argument('--do_remove_low_freq_words', type=int, default=5, metavar="RLW",
                        help='remove words in lower frequency')

    args = parser.parse_args()

    return args


def load_data(args, dataset_name):
    client_train_data_list = []
    if dataset_name == "20news":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.news_20.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.news_20.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_train_data_list.append(client_data_loader.get_train_batch_data(args.batch_size))
    elif dataset_name == "agnews":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.AGNews.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.AGNews.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_train_data_list.append(client_data_loader.get_train_batch_data(args.batch_size))
    elif dataset_name == "semeval_2010_task8":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.SemEval2010Task8.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.SemEval2010Task8.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_train_data_list.append(client_data_loader.get_train_batch_data(args.batch_size))
    elif dataset_name == "sentiment140":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.Sentiment140.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.Sentiment140.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_train_data_list.append(client_data_loader.get_train_batch_data(args.batch_size))
    elif dataset_name == "sst_2":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        server_data_loader = data_preprocessing.SST_2.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
        for client_index in range(server_data_loader.get_attributes()["n_clients"]):
            client_data_loader = data_preprocessing.SST_2.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                                 partition_method=args.partition_method, tokenize=True, client_idx=client_index)
            client_train_data_list.append(client_data_loader.get_train_batch_data(args.batch_size))
    else:
        raise Exception("No such dataset")
    return client_train_data_list, server_data_loader.get_test_batch_data(args.batch_size), \
           server_data_loader.get_attributes()


def preprocess_data(args, dataset):
    """
    preprocessing data for further training, which includes load pretrianed embeddings, padding data and transforming
    token and label to index
    """
    logging.info("preproccess data")
    train_batch_data_list, test_batch_data_list, attributes = dataset
    target_vocab = attributes["target_vocab"]

    # remove low frequency words and stop words
    # build frequency vocabulary based on tokenized data
    x = []
    for batch_data in train_batch_data_list:
        x.extend(batch_data["X"])
    for batch_data in test_batch_data_list:
        x.extend(batch_data["X"])
    freq_vocab = build_freq_vocab(x)
    logging.info("frequency vocab size %d", len(freq_vocab))

    if args.do_remove_low_freq_words > 0:
        logging.info("remove low frequency words")
        # build low frequency words set
        low_freq_words = set()
        for token, freq in freq_vocab.items():
            if freq <= args.do_remove_low_freq_words:
                low_freq_words.add(token)

        for i, batch_data in enumerate(train_batch_data_list):
            train_batch_data_list[i]["X"] = remove_words(batch_data["X"], low_freq_words)

        for i, batch_data in enumerate(test_batch_data_list):
            test_batch_data_list[i]["X"] = remove_words(batch_data["X"], low_freq_words)

    if args.do_remove_stop_words:
        logging.info("remove stop words")
        for i, batch_data in enumerate(train_batch_data_list):
            train_batch_data_list[i]["X"] = remove_words(batch_data["X"], STOP_WORDS)

        for i, batch_data in enumerate(test_batch_data_list):
            test_batch_data_list[i]["X"] = remove_words(batch_data["X"], STOP_WORDS)

    x.clear()
    x = []
    for batch_data in train_batch_data_list:
        x.extend(batch_data["X"])
    for batch_data in test_batch_data_list:
        x.extend(batch_data["X"])
    source_vocab = build_vocab(x)
    logging.info("source vocab size %d", len(source_vocab))

    # load pretrained embeddings. Note that we use source vocabulary here to reduce the input size
    embedding_weights = None
    if args.embedding_name:
        if args.embedding_name == "word2vec":
            logging.info("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_word2vec_embedding(os.path.abspath(args.embedding_file), source_vocab)
        elif args.embedding_name == "glove":
            logging.info("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_glove_embedding(os.path.abspath(args.embedding_file), source_vocab,
                                                                   args.embedding_length)
        else:
            raise Exception("No such embedding")
        embedding_weights = torch.tensor(embedding_weights, dtype=torch.float)

    if args.max_seq_len == -1:
        lengths = []
        for batch_data in train_batch_data_list:
            lengths.extend([len(single_x) for single_x in batch_data["X"]])
        args.max_seq_len = max(lengths)

    new_train_batch_data_list = list()
    new_test_batch_data_list = list()
    num_train_examples = 0
    num_test_examples = 0

    # padding data and transforming token as well as label to index
    for i, batch_data in enumerate(train_batch_data_list):
        padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
        new_train_batch_data_list.append(
            {"X": token_to_idx(padding_x, source_vocab),
             "Y": label_to_idx(batch_data["Y"], target_vocab),
             "seq_lens": seq_lens})
        num_train_examples += len(batch_data["X"])

    for batch_data in test_batch_data_list:
        padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
        new_test_batch_data_list.append(
            {"X": token_to_idx(padding_x, source_vocab),
             "Y": label_to_idx(batch_data["Y"], target_vocab),
             "seq_lens": seq_lens})
        num_test_examples += len(batch_data["X"])

    logging.info("number of train examples: %s, number of test examples: %s, size of source vocab: %s, "
          "size of target vocab: %s" % (num_train_examples, num_test_examples, len(source_vocab), len(target_vocab)))
    return new_train_batch_data_list, new_test_batch_data_list, source_vocab, target_vocab, embedding_weights


def create_model(args, model_name, input_size, output_size, embedding_weights):
    logging.info("create_model. model_name = %s, input_size = %s, output_size = %s"
          % (model_name, input_size, output_size))
    model = None
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


def FedNLP_text_classification_centralized(client_index, model, train_data, test_data, args):
    if args.device is not None:
        model = model.to(device=args.device)

    optimizer = None
    if args.optimizer == "adam":
        optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("No such optimizer")
    loss_func = F.cross_entropy
    max_eval_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_model(client_index, model, train_data, loss_func, optimizer, epoch, args)
        eval_loss, eval_acc = eval_model(model, test_data, loss_func, args)
        max_eval_acc = max(max_eval_acc, eval_acc)
        logging.info("Client index: %d, Epoch: %d, Train loss: %.4f, Train Accuracy: %.2f, Eval loss: %.4f, "
              "Eval Accuracy: %.2f" % (client_index, epoch + 1, train_loss, train_acc, eval_loss, eval_acc))
        wandb.log({"Epoch-Client %d" % client_index: epoch + 1, "Avg Training loss-Client %d" % client_index: train_loss,
                   "Avg Training Accuracy-Client %d" % client_index: train_acc,
                   "Avg Eval loss-Client %d" % client_index: eval_loss,
                   "Avg Eval Accuracy-Client %d" % client_index: eval_acc})
    return max_eval_acc


def train_model(client_index, model, train_data, loss_func, optimizer, epoch, args):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.train()
    steps = 0
    for batch_data in train_data:
        x = torch.tensor(batch_data["X"])
        y = torch.tensor(batch_data["Y"])
        seq_lens = torch.tensor(batch_data["seq_lens"])
        if args.device is not None:
            x = x.to(device=args.device)
            y = y.to(device=args.device)
            seq_lens = seq_lens.to(device=args.device)
        optimizer.zero_grad()
        prediction = model(x, x.size()[0], seq_lens, args.device)
        loss = loss_func(prediction, y)
        num_corrects = torch.sum(torch.argmax(prediction, 1) == y)
        acc = 100.0 * num_corrects / x.size()[0]
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 100 == 0:
            wandb.log({"Training loss-Client %d" % client_index: loss.item(),
                       "Training Accuracy-Client %d:" % client_index: acc.item()})
            logging.info("Client index: %d, Epoch: %d, Training loss: %.4f, Training Accuracy: %.2f" %
                  (client_index, epoch + 1, loss.item(), acc.item()))

        total_epoch_acc += acc.item()
        total_epoch_loss += loss.item()

    return total_epoch_loss / len(train_data), total_epoch_acc / len(train_data)


def eval_model(model, test_data, loss_func, args):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    for batch_data in test_data:
        x = torch.tensor(batch_data["X"])
        y = torch.tensor(batch_data["Y"])
        seq_lens = torch.tensor(batch_data["seq_lens"])
        if args.device is not None:
            x = x.to(device=args.device)
            y = y.to(device=args.device)
            seq_lens = seq_lens.to(device=args.device)
        prediction = model(x, x.size()[0], seq_lens, args.device)
        loss = loss_func(prediction, y)
        num_corrects = torch.sum(torch.argmax(prediction, 1) == y)
        acc = 100.0 * num_corrects / x.size()[0]

        total_epoch_acc += acc.item()
        total_epoch_loss += loss.item()

    return total_epoch_loss / len(test_data), total_epoch_acc / len(test_data)


if __name__ == "__main__":
    process_id = os.getpid()
    # customize the log format
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    logging.info("start")

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    if args.embedding_name:
        embedding_name = args.embedding_name
    else:
        embedding_name = "random"
    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    wandb.init(
        # project="federated_nas",
        project="fednlp",
        entity="automl",
        name="FedCentralized-LowerBound" + "-" + str(args.dataset) + "-" + str(args.model) + "-" + str(embedding_name) + "-e" +
             str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    client_train_data_list, test_batch_data_list, attributes = load_data(args, args.dataset)

    eval_accuracy_list = []
    for idx, train_batch_data_list in enumerate(client_train_data_list):
        dataset = preprocess_data(args, [train_batch_data_list, test_batch_data_list, attributes])

        # create model
        model = create_model(args, model_name=args.model, input_size=len(dataset[2]), output_size=len(dataset[3]),
                             embedding_weights=dataset[4])

        if not args.device:
            if torch.cuda.is_available():
                args.device = 'cuda'

        eval_acc = FedNLP_text_classification_centralized(idx, model, dataset[0], dataset[1], args)
        eval_accuracy_list.append(eval_acc)

    logging.info("%s lower bound test eval accuracy statistics" % args.dataset)
    logging.info("all eval accuracy %s" % ",".join([str(acc) for acc in eval_accuracy_list]))

    mean_accuracy = mean(eval_accuracy_list)
    logging.info("Mean eval accuracy: %.2f" % mean_accuracy)
    wandb.log({"Mean eval accuracy": mean_accuracy})

    max_accuracy = max(eval_accuracy_list)
    logging.info("Maximum eval accuracy: %.2f" % max_accuracy)
    wandb.log({"Maximum eval accuracy": max_accuracy})

    min_accuracy = min(eval_accuracy_list)
    logging.info("Minimum eval accuracy: %.2f" % min_accuracy)
    wandb.log({"Minimum eval accuracy": min_accuracy})

    median_accuracy = median(eval_accuracy_list)
    logging.info("Median eval accuracy: %.2f" % median_accuracy)
    wandb.log({"Median eval accuracy": median_accuracy})

    pvariance_accuracy = pvariance(eval_accuracy_list)
    logging.info("Pvariance of eval accuracy: %.2f" % pvariance_accuracy)
    wandb.log({"Pvariance of eval accuracy": pvariance_accuracy})

    pstdev_accuracy = pstdev(eval_accuracy_list)
    logging.info("Pstdev of eval accuracy: %.2f" % pstdev_accuracy)
    wandb.log({"Pstdev of eval accuracy": pstdev_accuracy})

    logging.info("end")
    
