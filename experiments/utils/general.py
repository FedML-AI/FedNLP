import torch
import numpy as np
import random
from model.transformer.model_args import ClassificationArgs
from model.transformer.bert_model import BertForSequenceClassification
from model.transformer.distilbert_model import DistilBertForSequenceClassification

from transformers import BertTokenizer, BertConfig

from training.tc_transformer_trainer import TextClassificationTrainer

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


def create_model(args, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "classification":{
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            },
        "sequence_tagging":{

        }, # TODO: add more.
        } 
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][args.model_type]
    config = config_class.from_pretrained(args.model_name, num_labels=args.num_labels, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config)
    tokenizer = tokenizer_class.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    # logging.info(self.model)
    return config, model, tokenizer


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_centralized_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Infrastructure related
    parser.add_argument('--device_id', type=int, default=8, metavar='N',  # TODO: why 8?
                        help='device id')

    # Data related
    # TODO: list all dataset names: 
    parser.add_argument('--dataset', type=str, default='agnews', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file_path', type=str, default='/home/bill/fednlp_data/data_files/agnews_data.h5',
                        help='data h5 file path')

    parser.add_argument('--partition_file_path', type=str,
                        default='/home/bill/fednlp_data/partition_files/agnews_partition.h5',
                        help='partition h5 file path')

    parser.add_argument('--partition_method', type=str, default='uniform',
                        help='partition method')

    # Model related
    parser.add_argument('--model_type', type=str, default='bert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
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

    parser.add_argument('--num_train_epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--evaluate_during_training_steps', type=int, default=100, metavar='EP',
                        help='the frequency of the evaluation during training')                        
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')
    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    return parser


def add_federated_args(parser): 
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser = add_centralized_args(parser)

    # Federated Learning related

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    return parser
