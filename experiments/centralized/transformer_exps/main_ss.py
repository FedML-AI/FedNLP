import argparse
import logging
import os
import sys

import numpy as np
import torch
# this is a temporal import, we will refactor FedML as a package installation
import wandb

wandb.init(mode="disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from data_preprocessing.seq2seq_preprocessor import TLMPreprocessor
from data_manager.seq2seq_data_manager import Seq2SeqDataManager

from model.transformer.model_args import Seq2SeqArgs
from training.ss_transformer_trainer import Seq2SeqTrainer
from experiments.centralized.transformer_exps.initializer import set_seed, add_centralized_args, create_model
 


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_centralized_args(parser) # add general args.
    # TODO: you can add customized args here.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    wandb.init(project="fednlp", entity="automl", name="FedNLP-Centralized" +
                                                "-SS-" + str(args.dataset) + "-" + str(args.model_name),
        config=args)

    # device
    device = torch.device("cuda:0")

    # attributes
    attributes = Seq2SeqDataManager.load_attributes(args.data_file_path)

    # model
    model_args = Seq2SeqArgs()    
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.fl_algorithm = ""
    model_args.use_multiprocessing = False
    # model_args.multiprocessing_chunksize = 10
    model_args.update_from_dict({"epochs": args.epochs,
                              "learning_rate": args.learning_rate,
                              "gradient_accumulation_steps": args.gradient_accumulation_steps,
                              "do_lower_case": args.do_lower_case,
                              "manual_seed": args.manual_seed,
                              "reprocess_input_data": args.reprocess_input_data, # True for ignoring the cache features.
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
                              "is_debug_mode": args.is_debug_mode,
                              "num_beams": 3
                              })
    model_config, model, tokenizer = create_model(model_args, formulation="seq2seq")

    # preprocessor
    preprocessor = TLMPreprocessor(args=model_args, tokenizer=tokenizer)

    # data manager
    process_id = 0
    num_workers = 1
    dm = Seq2SeqDataManager(args, model_args, preprocessor)
    train_dl, test_dl = dm.load_centralized_data() # cut_off = 1 for each client.


    

    # Create a Seq2Seq Trainer and start train
    trainer = Seq2SeqTrainer(model_args, device, model, train_dl, test_dl, tokenizer)
    trainer.train_model()
    trainer.eval_model()
    

''' Example Usage:

DATA_NAME=gigaword
CUDA_VISIBLE_DEVICES=6 python -m experiments.centralized.transformer_exps.main_ss \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type bart \
    --model_name facebook/bart-base  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --reprocess_input_data False \
    --evaluate_during_training_steps 1000 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1

'''