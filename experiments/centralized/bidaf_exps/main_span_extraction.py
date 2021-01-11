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

from torch import nn
import data_preprocessing.SQuAD_1_1.data_loader
from data_preprocessing.base.utils import *
from data_preprocessing.base.globals import *
from model.bidaf import BIDAF_SpanExtraction
from experiments.centralized.bidaf_exps.ema import EMA


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

    parser.add_argument('--hidden_size', type=int, default=100, metavar='H',
                        help='size of hidden layers')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='D',
                        help="dropout rate")

    parser.add_argument('--batch_size', type=int, default=60, metavar='B',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--max_sent_len', type=int, default=400, metavar='MSL',
                        help='maximum sentence length')
    
    parser.add_argument('--max_ques_len', type=int, default=30, metavar='MQL',
                        help='maximum question length')
    
    parser.add_argument('--max_word_len', type=int, default=16, metavar='MWL',
                        help='maximum word length')

    parser.add_argument('--word_emb_len', type=int, default=100, metavar="EL", help='dimension of word embedding')

    parser.add_argument('--char_emb_len', type=int, default=8, metavar="EL", help='dimension of character embedding')

    parser.add_argument('--word_emb_name', type=str, default="glove", metavar="WEN", help='word embedding name')

    parser.add_argument('--word_emb_file', type=str, default="data/pretrained/glove.6B.100d.txt", metavar="EF", help='word embedding file')

    parser.add_argument('--highway_num_layers', type=int, default=2, metavar="HNL", help='the number of layers in highway network')

    parser.add_argument('--out_channel_dims', type=str, default="100", metavar="OCD", help='Out channel dims of Char-CNN, separated by commas [100]')

    parser.add_argument('--filter_heights', type=str, default="5", metavar="FH", help='Filter heights of Char-CNN, separated by commas [5]')

    parser.add_argument('--min_token_freq', type=int, default=10, metavar="MTF", help='Minimum token frequency')

    parser.add_argument('--min_char_freq', type=int, default=50, metavar="MCF", help='Minimum character frequency')

    parser.add_argument('--char_out_size', type=int, default=100, metavar="COS", help='char-CNN output size')

    parser.add_argument('--optimizer', type=str, default='adadelta', metavar="O",
                        help='adam, adadelta')

    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 0.5)')

    parser.add_argument('--wd', help='weight decay parameter;', metavar="WD", type=float, default=0)

    parser.add_argument('--decay_rate', type=float, default=0.999, metavar='DR', help='decay rate for learning rate')

    parser.add_argument('--epochs', type=int, default=50, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--device', type=str, default="cuda:0", metavar="DV", help='gpu device for training')

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
    token_freq_vocab = build_freq_vocab(token_x)
    token_vocab = build_vocab(token_x)

    char_x = []
    for batch_data in train_batch_data_list:
        for sent in batch_data["char_context_X"]:
            char_x.extend(sent)
        for sent in batch_data["char_question_X"]:
            char_x.extend(sent)
    char_freq_vocab = build_freq_vocab(char_x)

    char_vocab = dict()
    for char, freq in char_freq_vocab.items():
        if freq > args.min_char_freq:
            char_vocab[char] = len(char_vocab)
    char_vocab[UNK_TOKEN] = len(char_vocab)
    char_vocab[PAD_TOKEN] = len(char_vocab)

    args.char_vocab_size = len(char_vocab)

    word_emb_weights = None
    if args.word_emb_name:
        logging.info("load word embedding %s" % args.word_emb_name)
        token_vocab, word_emb_weights = load_glove_embedding(os.path.abspath(args.word_emb_file), 
        token_vocab, args.word_emb_len)
        word_emb_weights = torch.tensor(word_emb_weights, dtype=torch.float32)
    
    new_token_vocab = dict()
    for token, freq in token_freq_vocab.items():
        if token not in token_vocab and freq > args.min_token_freq:
            new_token_vocab[token] = len(new_token_vocab)
    
    args.token_vocab_size = len(new_token_vocab)

    for token, idx in token_vocab.items():
        new_token_vocab[token] = idx + args.token_vocab_size
    
    token_vocab = new_token_vocab

    new_train_batch_data_list = list()
    new_test_batch_data_list = list()

    def process_batch_data_list(batch_data_list, new_batch_data_list):
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
            padding_context_chars, context_word_lens = padding_char_data(new_batch_data["cx"], args.max_sent_len, args.max_word_len)
            padding_question_chars, question_word_lens = padding_char_data(new_batch_data["cq"], args.max_ques_len, args.max_word_len)
            new_batch_data["x"] = np.array(token_to_idx(padding_context_tokens, token_vocab))
            new_batch_data["cx"] = np.array(char_to_idx(padding_context_chars, char_vocab))
            new_batch_data["q"] = np.array(token_to_idx(padding_question_tokens, token_vocab))
            new_batch_data["cq"] = np.array(char_to_idx(padding_question_chars, char_vocab))
            for j, context_len in enumerate(context_lens):
                new_batch_data["x_mask"][j, 0:context_len] = True 
            for j, question_len in enumerate(question_lens):
                new_batch_data["q_mask"][j, 0:question_len] = True 
            new_batch_data_list.append(new_batch_data)


    process_batch_data_list(train_batch_data_list, new_train_batch_data_list)
    process_batch_data_list(test_batch_data_list, new_test_batch_data_list)

    return [new_train_batch_data_list, new_test_batch_data_list, token_vocab, char_vocab, 
    word_emb_weights, attributes]

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
        


def create_model(args, model_name):
    logging.info("create_model. model_name = %s, token_vocab_size = %s, char_vocab_size = %s"
          % (model_name, args.token_vocab_size, args.char_vocab_size))
    model = None
    if model_name == "bidaf":
        model = BIDAF_SpanExtraction(args.char_vocab_size, args.token_vocab_size, args.char_emb_len, args.word_emb_len,
                                          args.out_channel_dims, args.filter_heights,
                                          args.max_word_len, args.char_out_size, args.dropout, args.hidden_size, 
                                          args.highway_num_layers)
    else:
        raise Exception("No such model")
    return model


def FedNLP_span_extraction_centralized(model, train_data, test_data, glove_emb_weights, args):
    if args.device is not None:
        model = model.to(device=args.device)

    ema = EMA(args.decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    optimizer = None
    if args.optimizer == "adadelta":
        optimizer = Adadelta(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "adam":
        optimizer = Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("No such optimizer")
    loss_func = F.cross_entropy
    max_eval_em = 0.0
    for epoch in range(args.epochs):
        train_loss, train_em = train_model(model, train_data, glove_emb_weights, loss_func, optimizer, epoch, ema, args)
        eval_loss, eval_em = eval_model(model, test_data, glove_emb_weights, loss_func, ema, args)
        max_eval_em = max(max_eval_em, eval_em)
        logging.info("Epoch: %d, Train loss: %.4f, Train Exact Match: %.2f, Eval loss: %.4f, Eval Exact Match: %.2f" % (epoch + 1,
                                                                                                           train_loss,
                                                                                                           train_em,
                                                                                                           eval_loss,
                                                                                                           eval_em))
        wandb.log({"Epoch": epoch + 1, "Avg Training loss": train_loss, "Avg Training Exact Match:": train_em,
                   "Avg Eval loss": eval_loss, "Avg Eval Exact Match": eval_em})
    logging.info("Maximum Eval Exact Match: %.2f" % max_eval_em)

def build_loss(logits, logits2, y, y2, loss_func):
    ce_loss = loss_func(logits, torch.argmax(y.long(), 1))
    ce_loss2 = torch.mean(loss_func(logits2, torch.argmax(y2.long(), 1)))
    return ce_loss + ce_loss2


def train_model(model, train_data, glove_emb_weights, loss_func, optimizer, epoch, ema, args):
    total_epoch_loss = 0
    total_epoch_em = 0
    model.train()
    steps = 0
    loss = 0
    for batch_data in train_data:
        x = torch.tensor(batch_data["x"])
        cx = torch.tensor(batch_data["cx"])
        x_mask = torch.tensor(batch_data["x_mask"])
        q = torch.tensor(batch_data["q"])
        cq = torch.tensor(batch_data["cq"])
        q_mask = torch.tensor(batch_data["q_mask"])
        y = torch.tensor(batch_data["y"])
        y2 = torch.tensor(batch_data["y2"])
        if args.device is not None:
            x = x.to(device=args.device)
            cx = cx.to(device=args.device)
            x_mask = x_mask.to(device=args.device)
            q = q.to(device=args.device)
            cq = cq.to(device=args.device)
            q_mask = q_mask.to(device=args.device)
            y = y.to(device=args.device)
            y2 = y2.to(device=args.device)
            glove_emb_weights = glove_emb_weights.to(device=args.device)
        optimizer.zero_grad()
        logits, logits2 = model(x, cx, x_mask, q, cq, q_mask, y, y2, glove_emb_weights, args.device)
        x_mask_float = x_mask.float()
        matched = [1 for l1, y1, l2, y2 in zip(torch.argmax(logits * x_mask_float, 1), torch.argmax(y.long(), 1), 
        torch.argmax(logits2 * x_mask_float, 1), torch.argmax(y2.long(), 1)) if l1 == y1 and l2 == y2]
        num_corrects = sum(matched)
        em = 100.0 * num_corrects / x.size()[0]
        batch_loss = build_loss(logits, logits2, y, y2, loss_func)
        loss += batch_loss
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)

        steps += 1
        if steps % 100 == 0:
            wandb.log({"Training loss": loss.item(), "Training Exact Match:": em})
            logging.info("Epoch: %d, Training loss: %.4f, Training Exact Match: %.2f" % (epoch + 1, loss.item(), em))
            loss = 0
        
        total_epoch_em += em
        total_epoch_loss += loss.item()

    return total_epoch_loss / len(train_data), total_epoch_em / len(train_data)

def eval_model(model, test_data, glove_emb_weights, loss_func, ema, args):
    total_epoch_loss = 0
    total_epoch_em = 0
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    for batch_data in test_data:
        x = torch.tensor(batch_data["x"])
        cx = torch.tensor(batch_data["cx"])
        x_mask = torch.tensor(batch_data["x_mask"])
        q = torch.tensor(batch_data["q"])
        cq = torch.tensor(batch_data["cq"])
        q_mask = torch.tensor(batch_data["q_mask"])
        y = torch.tensor(batch_data["y"])
        y2 = torch.tensor(batch_data["y2"])
        if args.device is not None:
            x = x.to(device=args.device)
            cx = cx.to(device=args.device)
            x_mask = x_mask.to(device=args.device)
            q = q.to(device=args.device)
            cq = cq.to(device=args.device)
            q_mask = q_mask.to(device=args.device)
            y = y.to(device=args.device)
            y2 = y2.to(device=args.device)
            glove_emb_weights = glove_emb_weights.to(device=args.device)
        logits, logits2 = model(x, cx, x_mask, q, cq, q_mask, y, y2, glove_emb_weights, args.device)
        x_mask_float = x_mask.float()
        matched = [1 for l1, y1, l2, y2 in zip(torch.argmax(logits * x_mask_float, 1), torch.argmax(y.long(), 1), 
        torch.argmax(logits2 * x_mask_float, 1), torch.argmax(y2.long(), 1)) if l1 == y1 and l2 == y2]
        num_corrects = sum(matched)
        em = 100.0 * num_corrects / x.size()[0]
        loss = build_loss(logits, logits2, y, y2, loss_func)
        
        total_epoch_em += em
        total_epoch_loss += loss.item()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_params.get(name))
    
    return total_epoch_loss / len(test_data), total_epoch_em / len(test_data)




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
    wandb.init(
        project="fednlp",
        entity="automl",
        name="FedCentralized" + "-" + str(args.dataset) + "-" + str(args.model) + "-" + str(word_emb_name) + "-e" +
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
    model = create_model(args, args.model)

    total_params = 0
    for name, param in model.named_parameters():
        print(name ,param.size())
        if param.requires_grad:
            total_params += param.numel()
    print("total parameters:", total_params)


    if not args.device:
        if torch.cuda.is_available():
            args.device = 'cuda'

    FedNLP_span_extraction_centralized(model, dataset[0], dataset[1], dataset[4], args)
    logging.info("end")
