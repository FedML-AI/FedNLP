import socket
import sys

import psutil
import os 
import numpy as np
import torch

# this is a temporal import, we will refactor FedML as a package installation
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from training.fed_trainer_transformer import FedTransformerTrainer
from experiments.utils.general import set_seed, create_model, add_federated_args
from data_preprocessing.seq_tagging_preprocessor import TLMPreprocessor
from training.st_transformer_trainer import SeqTaggingTrainer
from model.transformer.model_args import SeqTaggingArgs
from data_manager.base_data_manager import BaseDataManager
from data_manager.seq_tagging_data_manager import SequenceTaggingDataManager
from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed
import argparse
import logging
 

def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


# def load_data_to_FedML(model_args, args, process_id, num_workers, preprocessor):
#     # TODO: process_id should be 0 if we want to load all instances
#     dm = TextClassificationDataManager(
#         model_args, args, process_id, num_workers, preprocessor)
#     # train_data_num: {key: client_index; value: number of samples}
#     # train_data_global/xxx_data: {key: client_index; value: PyTorch-DataLoader}
#     return (train_data_num, train_data_global, test_data_global,
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict)


def fedavg_main(process_id, worker_number, device, args):

    # dataset attributes
    attributes = BaseDataManager.load_attributes(
        args.data_file_path)

    # model init
    model_args = SeqTaggingArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = len(attributes["label_vocab"])
    model_args.update_from_dict({"epochs": args.epochs,
                                 "learning_rate": args.learning_rate,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 # for ignoring the cache features.
                                 "reprocess_input_data": False,
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training": False, # Disabled for FedAvg.
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "is_debug_mode": args.is_debug_mode
                                 })

    model_config, client_model, tokenizer = create_model(
        model_args, formulation="seq_tagging")

    # data preprocessor
    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"],
        tokenizer=tokenizer)

    # data manager
    num_workers = args.client_num_per_round

    client_trainer = SeqTaggingTrainer(
        model_args, device, client_model, None, None, None, tokenizer)
    fed_trainer = FedTransformerTrainer(
        client_trainer, client_model, task_formulation="seq_tagging")
    dm = SequenceTaggingDataManager(args, model_args, preprocessor, process_id, num_workers)
    train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \
        train_data_local_dict, test_data_local_dict, num_clients = dm.load_federated_data(process_id=process_id)
    # start FedAvg algorithm
    # for distributed algorithm, train_data_gloabl and test_data_global are required
    args.client_num_in_total = num_clients
    FedML_FedAvg_distributed(
        process_id, worker_number, device, comm, client_model, train_data_num,
        train_data_global, test_data_global, train_data_local_num_dict,
        train_data_local_dict, test_data_local_dict, args, fed_trainer)


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_federated_args(parser)
    # TODO: add more customized args.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
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

    # logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    if process_id == 0:
        # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
        wandb.init(
            project="fednlp", entity="automl", name="FedNLP-FedAVG-Transformer" +
                                                    "-TC-" + str(args.dataset) + "-" + str(args.model_name),
            config=args)

    # device: check "gpu_mapping.yaml" to see how to define the topology
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)
    logging.info("process_id = %d, size = %d, device=%s" %
                 (process_id, worker_number, str(device)))

    logging.info("torch.cuda.current_device()=" + str(torch.cuda.current_device()))
    logging.info("torch.cuda.device_count()=" + str(torch.cuda.device_count()))

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

