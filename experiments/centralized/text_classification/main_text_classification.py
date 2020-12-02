import argparse
import os
import random

import torch
import torch.nn.functional as F
import wandb
from torch.optim import *

import data_preprocessing.AGNews.data_loader
import data_preprocessing.SST_2.data_loader
import data_preprocessing.SemEval2010Task8.data_loader
import data_preprocessing.Sentiment140.data_loader
import data_preprocessing.news_20.data_loader
from data_preprocessing.base.utils import *
from model.bilstm import BiLSTM


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='bilstm', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='20news', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/20news_data_loader.pkl',
                        help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='data/partition/20news_partition.pkl',
                        help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
                        help='how to partition the dataset')

    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='size of hidden layers')

    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of layers in neural network')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                        help='dropout rate for neural network')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='N',
                        help='maximum sequence length (-1 means the maximum sequence length in the dataset)')

    parser.add_argument('--embedding_file', type=str, nargs="?", help='word embedding file')

    parser.add_argument('--embedding_name', type=str, nargs="?", help='word embedding name(word2vec, glove)')

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
        print("load_data. dataset_name = %s" % dataset_name)
        data_loader = data_preprocessing.news_20.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "agnews":
        print("load_data. dataset_name = %s" % dataset_name)
        data_loader = data_preprocessing.AGNews.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "semeval_2010_task8":
        print("load_data. dataset_name = %s" % dataset_name)
        data_loader = data_preprocessing.SemEval2010Task8.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "sentiment140":
        print("load_data. dataset_name = %s" % dataset_name)
        data_loader = data_preprocessing.Sentiment140.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    elif dataset_name == "sst_2":
        print("load_data. dataset_name = %s" % dataset_name)
        data_loader = data_preprocessing.SST_2.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True)
    else:
        raise Exception("No such dataset")
    dataset = [data_loader.get_train_batch_data(args.batch_size), data_loader.get_test_batch_data(args.batch_size),
               data_loader.get_attributes()]
    return dataset


def preprocess_data(args, dataset):
    print("preproccess data")
    train_batch_data_list, test_batch_data_list, attributes = dataset
    target_vocab = attributes["target_vocab"]
    x = []
    for batch_data in train_batch_data_list:
        x.extend(batch_data["X"])
    source_vocab = build_vocab(x)
    embedding_weights = None
    if args.embedding_name is not None:
        if args.embedding_name == "word2vec":
            print("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_word2vec_embedding(os.path.abspath(args.embedding_file),
                                                                      source_vocab)
        elif args.embedding_name == "glove":
            print("load word embedding %s" % args.embedding_name)
            source_vocab, embedding_weights = load_glove_embedding(os.path.abspath(args.embedding_file))
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
    for i, batch_data in enumerate(train_batch_data_list):
        new_train_batch_data_list.append(
            {"X": token_to_idx(padding_data(batch_data["X"], args.max_seq_len), source_vocab),
             "Y": label_to_idx(batch_data["Y"], target_vocab)})
        num_train_examples += len(batch_data["X"])

    for batch_data in test_batch_data_list:
        new_test_batch_data_list.append(
            {"X": token_to_idx(padding_data(batch_data["X"], args.max_seq_len), source_vocab),
             "Y": label_to_idx(batch_data["Y"], target_vocab)})
        num_test_examples += len(batch_data["X"])

    print("number of train examples: %s, number of test examples: %s, size of source vocab: %s, "
          "size of target vocab: %s" % (num_train_examples, num_test_examples, len(source_vocab), len(target_vocab)))
    return new_train_batch_data_list, new_test_batch_data_list, source_vocab, target_vocab, embedding_weights


def create_model(args, model_name, input_size, output_size, embedding_weights):
    print("create_model. model_name = %s, input_size = %s, output_size = %s"
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


def FedNLP_text_classification_centralized(model, train_data, test_data, args):
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
        train_loss, train_acc = train_model(model, train_data, loss_func, optimizer, epoch, args)
        eval_loss, eval_acc = eval_model(model, test_data, loss_func, args)
        max_eval_acc = max(max_eval_acc, eval_acc)
        print("Epoch: %d, Train loss: %.4f, Train Accuracy: %.2f, Eval loss: %.4f, Eval Accuracy: %.2f" % (epoch + 1,
                                                                                                           train_loss,
                                                                                                           train_acc,
                                                                                                           eval_loss,
                                                                                                           eval_acc))
        wandb.log({"Epoch": epoch + 1, "Avg Training loss": train_loss, "Avg Training Accuracy:": train_acc,
                   "Avg Eval loss": eval_loss, "Avg Eval Accuracy": eval_acc})
    print("Maximum Eval Accuracy: %.2f" % max_eval_acc)


def train_model(model, train_data, loss_func, optimizer, epoch, args):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.train()
    steps = 0
    for batch_data in train_data:
        x = torch.tensor(batch_data["X"])
        y = torch.tensor(batch_data["Y"])
        if args.device is not None:
            x = x.to(device=args.device)
            y = y.to(device=args.device)
        optimizer.zero_grad()
        prediction = model(x, x.size()[0], args.device)
        loss = loss_func(prediction, y)
        num_corrects = torch.sum(torch.argmax(prediction, 1) == y)
        acc = 100.0 * num_corrects / x.size()[0]
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 100 == 0:
            wandb.log({"Training loss": loss.item(), "Training Accuracy:": acc.item()})
            print("Epoch: %d, Training loss: %.4f, Training Accuracy: %.2f" % (epoch + 1, loss.item(), acc.item()))

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
        if args.device is not None:
            x = x.to(device=args.device)
            y = y.to(device=args.device)
        prediction = model(x, x.size()[0], args.device)
        loss = loss_func(prediction, y)
        num_corrects = torch.sum(torch.argmax(prediction, 1) == y)
        acc = 100.0 * num_corrects / x.size()[0]

        total_epoch_acc += acc.item()
        total_epoch_loss += loss.item()

    return total_epoch_loss / len(test_data), total_epoch_acc / len(test_data)


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)

    if args.embedding_name is None:
        embedding_name = "random"
    else:
        embedding_name = args.embedding_name
    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    wandb.init(
        # project="federated_nas",
        project="fednlp",
        name="FedCentralized" + "-" + str(args.dataset) + "-" + str(args.model) + "-" + str(embedding_name) + "-e" +
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
    dataset = load_data(args, args.dataset)
    dataset = preprocess_data(args, dataset)

    # create model
    model = create_model(args, model_name=args.model, input_size=len(dataset[2]), output_size=len(dataset[3]),
                         embedding_weights=dataset[4])

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'

    FedNLP_text_classification_centralized(model, dataset[0], dataset[1], args)
