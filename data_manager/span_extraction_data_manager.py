import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from data_manager.base_data_manager import BaseDataManager
from tqdm import tqdm
import logging

class SpanExtractionDataManager(BaseDataManager):
    """Data manager for reading comprehension (span-based QA).""" 
    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)
        super(SpanExtractionDataManager, self).__init__(args, model_args, process_id, num_workers)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

        
    def read_instance_from_h5(self, data_file, index_list):
        context_X = list()
        question_X = list()
        y = list()
        qas_ids = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file."):
            context_X.append(data_file["context_X"][str(idx)][()].decode("utf-8"))
            question_X.append(data_file["question_X"][str(idx)][()].decode("utf-8"))
            y.append(data_file["Y"][str(idx)][()])
            if "question_ids" in data_file:
                qas_ids.append(data_file["question_ids"][str(idx)][()].decode("utf-8"))
        return {"context_X": context_X, "question_X": question_X, "y": y, "qas_ids": qas_ids if qas_ids else None}

if __name__ == "__main__":
    from experiments.utils.general import create_model, add_federated_args
    from data_preprocessing.span_extraction_preprocessor import TLMPreprocessor
    from model.transformer.model_args import SpanExtractionArgs
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    args = add_federated_args(parser)
    args = parser.parse_args()
    # model init
    model_args = SpanExtractionArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
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
        model_args, formulation="span_extraction")

    # data preprocessor
    preprocessor = TLMPreprocessor(args=model_args, tokenizer=tokenizer)
    for process_id in range(1, 11):
        print(process_id)
        dm = SpanExtractionDataManager(args, model_args, preprocessor, process_id, 10)
        train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \
            train_data_local_dict, test_data_local_dict, num_clients = dm.load_federated_data(process_id=process_id)
    
'''
WORKER_NUM=10
ROUND=10
CI=0
DATA_DIR=../data/fednlp_data/
DATA_NAME=squad_1.1
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
python -m span_extraction_data_manager \
    --gpu_mapping_file "gpu_mapping.yaml" \
    --gpu_mapping_key "mapping_lambda-server2" \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset "${DATA_NAME}" \
    --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
    --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --epochs 1 \
    --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
    --fp16
'''