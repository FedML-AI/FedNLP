import argparse
import os
import random
import logging
import functools

import torch
import torch.nn.functional as F
import wandb
from torch.optim import *
import numpy as np


import data_preprocessing.SQuAD_1_1.data_loader
from data_preprocessing.base.utils import *
from data_preprocessing.base.globals import *
from model.bidaf import BIDAF_SpanExtraction


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='bidaf', metavar='M',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='squad1.1', metavar='DS',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/squad_1.1_data_loader.pkl',
                        metavar="DF", help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='data/partition/squad_1.1_partition.pkl',
                        metavar="PF", help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='PM',
                        help='how to partition the dataset')

    parser.add_argument('--hidden_size', type=int, default=300, metavar='H',
                        help='size of hidden layers')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='D',
                        help="dropout rate")

    parser.add_argument('--batch_size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--max_sent_len', type=int, default=400, metavar='MSL',
                        help='maximum sentence length')
    
    parser.add_argument('--max_ques_len', type=int, default=30, metavar='MQL',
                        help='maximum question length')
    
    parser.add_argument('--max_word_len', type=int, default=16, metavar='MWL',
                        help='maximum word length')

    parser.add_argument('--word_emb_len', type=int, default=100, metavar="EL", help='dimension of word embedding')

    parser.add_argument('--char_emb_len', type=int, default=100, metavar="EL", help='dimension of character embedding')

    parser.add_argument('--use_char_emb', type=bool, default=True, metavar="UCE", help='whether use char embedding')

    parser.add_argument('--use_word_emb', type=bool, default=True, metavar="UWE", help='whether use word embedding')

    parser.add_argument('--use_glove_for_unk', type=bool, default=True, metavar="UCE", help='whether use glove embedding for unkown tokens')

    parser.add_argument('--glove_emb_file', type=str, default="data/pretrained/glove.6B.100d.txt", metavar="EF", help='character embedding file')
    
    parser.add_argument('--use_highway_network', type=bool, default=True, metavar="UCE", help='whether use highway network')

    parser.add_argument('--optimizer', type=str, default='adam', metavar="O",
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--device', type=str, default="cuda:3", metavar="DV", help='gpu device for training')

    args = parser.parse_args()

    return args


def load_data(args, dataset_name):
    data_loader = None
    if dataset_name == "squad1.1":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        data_loader = data_preprocessing.SQuAD_1_1.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                             partition_method=args.partition_method, tokenize=True, 
                             data_filter=functools.partial(data_filter, args=args))
    else:
        raise Exception("No such dataset")
    dataset = [data_loader.get_train_batch_data(args.batch_size), data_loader.get_test_batch_data(args.batch_size),
               data_loader.get_attributes()]
    return dataset


def preprocess_data(args, dataset):
    """
    preprocessing data for further training, which includes load pretrianed embeddings, padding data and transforming
    token and label to index
    """
    logging.info("preproccess data")
    train_batch_data_list, test_batch_data_list, attributes = dataset
    doc_index = attributes["doc_index"]
    para_index = attributes["para_index"]

    token_x = []
    for batch_data in train_batch_data_list:
        token_x.extend(batch_data["tokenized_context_X"])
        token_x.extend(batch_data["tokenized_question_X"])
    token_vocab = build_vocab(token_x)

    char_x = []
    for batch_data in train_batch_data_list:
        for sent in batch_data["char_context_X"]:
            char_x.extend(sent)
        for sent in batch_data["char_question_X"]:
            char_x.extend(sent)
    char_vocab = build_vocab(char_x)

    addtional_word_emb_weights = None
    if args.use_glove_for_unk:
        logging.info("load word embedding glove")
        addtional_token_vocab, addtional_word_emb_weights = load_glove_embedding(os.path.abspath(args.glove_emb_file), 
        None, args.word_emb_len)
        addtional_word_emb_weights = np.array(addtional_word_emb_weights, dtype=np.float32)

    new_train_batch_data_list = list()
    new_test_batch_data_list = list()

    def _token_to_idx(x, test):
        idx_x = []
        for single_x in x:
            new_single_x = []
            for token in single_x:
                if token in token_vocab:
                    new_single_x.append(token_vocab[token])
                else:
                    if test and args.use_glove_for_unk:
                        if token in addtional_token_vocab:
                            new_single_x.append(addtional_token_vocab[token] + len(token_vocab))
                            continue
                    new_single_x.append(token_vocab[UNK_TOKEN])
            idx_x.append(new_single_x)
        return idx_x

    def process_batch_data_list(batch_data_list, new_batch_data_list, test):
        for batch_data in batch_data_list:
            new_batch_data = dict()
            batch_size = len(batch_data["context_X"])
            new_batch_data["x"] = list()
            new_batch_data["cx"] = list()
            new_batch_data["x_mask"] = np.zeros([batch_size, args.max_sent_len], dtype="bool")
            new_batch_data["q"] = list()
            new_batch_data["cq"] = list()
            new_batch_data["q_mask"] = np.zeros([batch_size, args.max_ques_len], dtype="bool")
            new_batch_data["y"] = np.zeros([batch_size, args.max_sent_len], dtype="bool")
            new_batch_data["y2"] = np.zeros([batch_size, args.max_sent_len], dtype="bool")
            for i in range(batch_size):
                context = batch_data["context_X"][i]
                context_tokens = batch_data["tokenized_context_X"][i]
                question_tokens = batch_data["tokenized_question_X"][i]
                yi0, yi1 = batch_data["Y"][i]
                context_chars = batch_data["char_context_X"][i]
                question_chars = batch_data["char_question_X"][i]
                new_batch_data["x"].append(context_tokens)
                new_batch_data["cx"].append(context_chars)
                new_batch_data["q"].append(question_tokens)
                new_batch_data["cq"].append(question_chars)
                new_batch_data["y"][i][yi0] = True
                new_batch_data["y2"][i][yi1] = True
            padding_context_tokens, context_lens = padding_data(new_batch_data["x"], args.max_sent_len)
            padding_question_tokens, question_lens = padding_data(new_batch_data["q"], args.max_ques_len)
            padding_context_chars, context_word_lens = padding_char_data(new_batch_data["cx"], args.max_word_len)
            padding_question_chars, question_word_lens = padding_char_data(new_batch_data["cq"], args.max_word_len)
            new_batch_data["x"] = np.array(_token_to_idx(padding_context_tokens, test))
            new_batch_data["cx"] = np.array(char_to_idx(padding_context_chars, char_vocab))
            new_batch_data["q"] = np.array(_token_to_idx(padding_question_tokens, test))
            new_batch_data["cq"] = np.array(char_to_idx(padding_question_chars, char_vocab))
            for j, context_len in enumerate(context_lens):
                new_batch_data["x_mask"][i][:context_len] = True 
            for j, question_len in enumerate(question_lens):
                new_batch_data["q_mask"][i][:question_len] = True 
            new_batch_data_list.append(new_batch_data)


    process_batch_data_list(train_batch_data_list, new_train_batch_data_list, False)
    process_batch_data_list(test_batch_data_list, new_test_batch_data_list, True)

def data_filter(data, args):
    removed_indices = list()
    for i in range(len(data["tokenized_question_X"])):
        if len(data["tokenized_question_X"][i]) > args.max_ques_len:
            removed_indices.append(i)
        elif data["Y"][i][1] > args.max_sent_len:
            removed_indices.append(i)
    for idx in removed_indices[::-1]:
        for key in data.keys():
            del data[key][idx]
        


def create_model(args, model_name, input_size, output_size, embedding_weights):
    logging.info("create_model. model_name = %s, input_size = %s, output_size = %s"
          % (model_name, input_size, output_size))
    model = None
    if model_name == "bidaf":
        model = BIDAF_SpanExtraction(input_size, args.hidden_size, output_size, args.num_layers,
                                          args.embedding_dropout, args.lstm_dropout, args.attention_dropout,
                                          args.embedding_length, attention=True, embedding_weights=embedding_weights)
    else:
        raise Exception("No such model")
    return model


def FedNLP_span_extraction_centralized(model, train_data, test_data, args):
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
        logging.info("Epoch: %d, Train loss: %.4f, Train Accuracy: %.2f, Eval loss: %.4f, Eval Accuracy: %.2f" % (epoch + 1,
                                                                                                           train_loss,
                                                                                                           train_acc,
                                                                                                           eval_loss,
                                                                                                           eval_acc))
        wandb.log({"Epoch": epoch + 1, "Avg Training loss": train_loss, "Avg Training Accuracy:": train_acc,
                   "Avg Eval loss": eval_loss, "Avg Eval Accuracy": eval_acc})
    logging.info("Maximum Eval Accuracy: %.2f" % max_eval_acc)


def train_model(model, train_data, loss_func, optimizer, epoch, args):
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
            wandb.log({"Training loss": loss.item(), "Training Accuracy:": acc.item()})
            logging.info("Epoch: %d, Training loss: %.4f, Training Accuracy: %.2f" % (epoch + 1, loss.item(), acc.item()))

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


    word_emb_name = "random"
    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    # wandb.init(
    #     # project="federated_nas",
    #     project="fednlp",
    #     entity="automl",
    #     name="FedCentralized" + "-" + str(args.dataset) + "-" + str(args.model) + "-" + str(word_emb_name) + "-e" +
    #          str(args.epochs) + "-lr" + str(args.lr),
    #     config=args
    # )

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

    if not args.device:
        if torch.cuda.is_available():
            args.device = 'cuda'

    FedNLP_span_extraction_centralized(model, dataset[0], dataset[1], args)
    logging.info("end")
