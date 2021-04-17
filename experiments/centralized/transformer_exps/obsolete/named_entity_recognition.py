"""
An example of running centralized experiments of fed-transformer models in FedNLP.
Example usage: 
(under the root folder)
CUDA_VISIBLE_DEVICES=2 python -m experiments.centralized.transformer_exps.named_entity_recognition \
    --dataset wikigold \
    --data_file data/data_files/wikigold_data.h5 \
    --partition_file data/partition_file/wikigold_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case False \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 128 \
    --learning_rate 4e-5 \
    --epochs 2 \
    --output_dir /tmp/wikigold/ \
    --fp16
"""
import data_preprocessing.W_NUT.data_loader
import data_preprocessing.wikigold.data_loader
from data_preprocessing.base.utils import *
from model.fed_transformers.ner import NERModel
import pandas as pd
import logging
import sklearn
import argparse
import wandb


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Data related
    parser.add_argument('--dataset', type=str, default='squad_1.1', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default='data/data_loaders/squad_1.1_data_loader.pkl',
                        help='data pickle file')

    parser.add_argument('--partition_file', type=str, default='data/partition/squad_1.1_partition.pkl',
                        help='partition pickle file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
                        help='how to partition the dataset')

    # Model related
    parser.add_argument('--model_type', type=str, default='distilbert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=False, metavar='N',
                        help='transformer model name')

    # Learning related

    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--learning_rate', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')
    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO realted

    parser.add_argument('--output_dir', type=str, default="/tmp/squad_1.1", metavar='N',
                        help='path to save the trained results and ckpts')

    args = parser.parse_args()

    return args


def load_data(args, dataset):
    data_loader = None
    print("Loading dataset = %s" % dataset)
    assert dataset in ["w_nut", "wikigold"]
    if dataset == "w_nut":
        data_loader = data_preprocessing.W_NUT.data_loader.ClientDataLoader(
            args.data_file, args.partition_file, partition_method=args.partition_method, tokenize=False) 
    elif dataset == "wikigold":
        data_loader = data_preprocessing.wikigold.data_loader.ClientDataLoader(
            args.data_file, args.partition_file, partition_method=args.partition_method, tokenize=False) 
    return data_loader.get_train_batch_data(), data_loader.get_test_batch_data(), data_loader.get_attributes()

def main(args):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Loading full data (for centralized learning)
    train_data, test_data, _ = load_data(args, args.dataset) 
    
    train_data = NER_data_formatter(train_data)
    test_data = NER_data_formatter(test_data)  
    
    print(train_data[:30])
    print(len(train_data), train_data[-1][0])

    train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])
    test_df = pd.DataFrame(test_data, columns=["sentence_id", "words", "labels"])
 
    # Create a ClassificationModel.
    model = NERModel(
        args.model_type, args.model_name, 
        args={"epochs": args.epochs,
              "learning_rate": args.learning_rate,
              "gradient_accumulation_steps": args.gradient_accumulation_steps,
              "do_lower_case": args.do_lower_case,
              "manual_seed": args.manual_seed,
              "reprocess_input_data": True,
              "overwrite_output_dir": True,
              "max_seq_length": args.max_seq_length,
              "train_batch_size": args.train_batch_size,
              "eval_batch_size": args.eval_batch_size,
              "fp16": args.fp16,
              "n_gpu": args.n_gpu,
              "output_dir": args.output_dir,
              "dataloader_num_workers": 1,
              "thread_count": 1, 
              "use_multiprocessing": False,
              "process_count": 1, 
              "wandb_project": "fednlp"})

    # Strat training.
    model.train_model(train_df)

    # Evaluate the model 
    result, model_outputs, preds_list = model.eval_model(test_df, output_dir=args.output_dir, verbose=False, silent=False)
    print(result) 


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)
    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    wandb.init(
        project="fednlp",
        entity="automl",
        name="FedNLP-Centralized" + "-NER-" + str(args.dataset) + "-" + str(args.model_name),
        config=args)
    # Start training.
    main(args)
