import argparse
import logging
import os
import sys

import torch
# this is a temporal import, we will refactor FedML as a package installation
import wandb

wandb.init(mode="disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from data_preprocessing.span_extraction_preprocessor import TLMPreprocessor
from data_manager.span_extraction_data_manager import SpanExtractionDataManager

from model.transformer.model_args import SpanExtractionArgs

from training.se_transformer_trainer import SpanExtractionTrainer

from experiments.centralized.transformer_exps.initializer import set_seed, add_centralized_args, create_model

if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_centralized_args(parser)  # add general args.
    # TODO: you can add customized args here.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # device
    print("device count:",torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:"+",".join([str(i) for i in range(args.n_gpu)]))    
    print(device)

    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    wandb.init(project="fednlp", entity="automl", name="FedNLP-Centralized" +
                                                "-SE-" + str(args.dataset) + "-" + str(args.model_name),
        config=args)

    # attributes
    attributes = SpanExtractionDataManager.load_attributes(args.data_file_path)

    # model
    model_args = SpanExtractionArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    # temporary add
    model_args.fl_algorithm = ""
    model_args.update_from_dict({"epochs": args.epochs,
                                 "learning_rate": args.learning_rate,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 "reprocess_input_data": args.reprocess_input_data,  # for ignoring the cache features.
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
                                 "n_gpu": args.n_gpu
                                 })

    model_config, model, tokenizer = create_model(model_args, formulation="span_extraction")

    # preprocessor
    preprocessor = TLMPreprocessor(args=model_args, tokenizer=tokenizer)

    # data manager
    dm = SpanExtractionDataManager(args, model_args, preprocessor)

    # dm._load_federated_data_server()

    train_dl, test_dl = dm.load_centralized_data()

    # Create a SpanExtractionModel and start train
    trainer = SpanExtractionTrainer(model_args, device, model, train_dl, test_dl, tokenizer)
    trainer.train_model()
    trainer.eval_model()

''' Example Usage:

DATA_NAME=mrqa
CUDA_VISIBLE_DEVICES=0 python -m experiments.centralized.transformer_exps.main_se \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --evaluate_during_training_steps 200 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1

'''
