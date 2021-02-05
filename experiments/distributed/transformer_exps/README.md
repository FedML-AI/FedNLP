```
pip install setproctitle mpi4py
pip install paho-mqtt 
```

# FedAvg for Transformer-based Text Classifcation

```bash
LOG_FILE="experiments/distributed/transformer_exps/fedavg_transformer_tc.log"
CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
ROUND=500
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
HOST_FILE=experiments/distributed/transformer_exps/mpi_host_file
hostname > $HOST_FILE

mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m experiments.distributed.transformer_exps.main_text_classification \
    --gpu_mapping_file "experiments/distributed/transformer_exps/gpu_mapping.yaml" \
    --gpu_mapping_key mapping_ink-ron \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset 20news \
    --data_file "data/data_files/20news_data.h5" \
    --partition_file "data/partition_files/20news_partition.h5" \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --server_lr 1e-5 \
    --server_optimizer admn \
    --epochs 1 \
    --output_dir "/tmp/20news_fedavg/" \
    --fed_alg fedavg \
    --fp16 
    # 2> ${LOG_FILE} &
```



# FedAvg for Transformer-based Question Answering (reading comprehension)

```bash
LOG_FILE="experiments/distributed/transformer_exps/fedavg_transformer_qa.log"
CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
ROUND=500
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
HOST_FILE=experiments/distributed/transformer_exps/mpi_host_file
hostname > $HOST_FILE

mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m experiments.distributed.transformer_exps.main_question_answering \
    --gpu_mapping_file "experiments/distributed/transformer_exps/gpu_mapping.yaml" \
    --gpu_mapping_key mapping_ink-ron \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset squad_1.1 \
    --data_file data/data_files/squad_1.1_data.h5 \
    --partition_file data/partition_files/squad_1.1_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 1e-5 \
    --server_lr 1e-5 \
    --server_optimizer adam \
    --epochs 1 \
    --output_dir "/tmp/squad_fedavg/" \
    --fed_alg fedavg \
    --fp16 
    # 2> ${LOG_FILE} &
```


# FedAvg for Transformer-based Sequence Tagging (e.g., NER)

```bash
LOG_FILE="experiments/distributed/transformer_exps/fedavg_transformer_qa.log"
CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
ROUND=500
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
HOST_FILE=experiments/distributed/transformer_exps/mpi_host_file
hostname > $HOST_FILE

mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m experiments.distributed.transformer_exps.sequence_tagging_fedavg \
    --gpu_mapping_file "experiments/distributed/transformer_exps/gpu_mapping.yaml" \
    --gpu_mapping_key mapping_ink-ron \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset wikigold \
    --data_file data/data_files/wikigold_data.h5 \
    --partition_file data/partition_files/wikigold_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --epochs 1 \
    --output_dir "/tmp/wikigold_fedavg/" \
    --fp16 
    # 2> ${LOG_FILE} &
```



```
nohup mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m experiments.distributed.transformer_exps.main_text_classification \
    --gpu_mapping_file "experiments/distributed/transformer_exps/gpu_mapping.yaml" \
    --gpu_mapping_key mapping_ink-ron \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset 20news \
    --data_file "data/data_files/20news_data.h5" \
    --partition_file "data/partition_files/20news_partition.h5" \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --server_lr 1e-5 \
    --server_optimizer admn \
    --epochs 1 \
    --output_dir "/tmp/20news_fedavg/" \
    --fed_alg fedavg \
    --fp16 > ${LOG_FILE} 2>&1 &
```
