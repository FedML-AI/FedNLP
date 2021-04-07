import argparse
import logging
import os
import random
import sys
import socket
import sys

import psutil

import numpy as np
import torch

# this is a temporal import, we will refactor FedML as a package installation
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed
from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from data_manager.text_classification_data_manager import TextClassificationDataManager
from model.transformer.model_args import ClassificationArgs
from training.tc_transformer_trainer import TextClassificationTrainer

from data_preprocessing.text_classification_preprocessor import TLMPreprocessor

from experiments.utils.general import set_seed, create_model, add_federated_args
from training.fed_trainer_transformer import FedTransformerTrainer



def create_model(args, num_labels):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name, num_labels=num_labels, **args.config)
    model = model_class.from_pretrained(model_name, config=config)
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=args.do_lower_case)
    # logging.info(self.model)
    return config, model, tokenizer


def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


def load_data_to_FedML():
    dm.load_next_round_data()  # The centralized version.
    train_dl, test_dl = dm.get_data_loader()
    test_examples = dm.test_examples
    return train_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict

def fedavg_main(process_id, worker_number, device, args):
    
    # dataset attributes
    attributes = TextClassificationDataManager.load_attributes(args.data_file_path)


    # model init
    model_args = ClassificationArgs()    
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = len(attributes["label_vocab"])
    model_args.update_from_dict({"num_train_epochs": args.num_train_epochs,
                              "learning_rate": args.learning_rate,
                              "gradient_accumulation_steps": args.gradient_accumulation_steps,
                              "do_lower_case": args.do_lower_case,
                              "manual_seed": args.manual_seed,
                              "reprocess_input_data": True, # for ignoring the cache features.
                              "overwrite_output_dir": True,
                              "max_seq_length": args.max_seq_length,
                              "train_batch_size": args.train_batch_size,
                              "eval_batch_size": args.eval_batch_size,
                              "evaluate_during_training_steps": args.evaluate_during_training_steps,
                              "fp16": args.fp16,
                              "data_file_path": args.data_file_path,
                              "partition_file_path": args.partition_file_path,
                              "partition_method": args.partition_method,
                              "dataset": args.dataset,
                              "output_dir": args.output_dir,
                              "is_debug_mode": args.is_debug_mode
                              })


    model_config, client_model, tokenizer = create_model(model_args, formulation="classification")

    # data preprocessor
    preprocessor = TLMPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)

    # data manager
    num_workers = 1
    dm = TextClassificationDataManager(model_args, args, process_id, num_workers, preprocessor)
    
 
    
    client_trainer = TextClassificationTrainer(model_args, device, client_model, None, None, None)
    fed_trainer = FedTransformerTrainer(client_trainer, client_model, task_formulation="classification")

    train_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict = load_data_to_FedML(dm)
    # start FedAvg algorithm
    # for distributed algorithm, train_data_gloabl and test_data_global are required
    FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                             client_model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,
                             fed_trainer)


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_federated_args(parser)
    # TODO: add more customized args.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    if process_id == 0:
        # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
        wandb.init(
            project="fednlp", entity="automl", name="FedNLP-FedAVG-Transformer" +
                                                    "-TC-" + str(args.dataset) + "-" + str(args.model_name),
            config=args)

    # device: check "gpu_mapping.yaml" to see how to define the topology
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, \
                                                            args.gpu_mapping_file, args.gpu_mapping_key)
    logging.info("process_id = %d, size = %d, device=%s" % (process_id, worker_number, str(device)))

    fedavg_main(process_id, worker_number, device, args)


    # TODO: add the wandb later.
    # if args.global_rank == 0:
    #     run = wandb.init(
    #         project="fednlp", entity="automl", name="FedNLP-FedAVG-Transformer" +
    #                                                 "-TC-" + str(args.dataset) + "-" + str(args.model_name),
    #         config=args)
    #     run.finish()

    if args.local_rank == 0:
        post_complete_message(args)

    