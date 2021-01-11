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
python -m experiments.distributed.transformer_exps.text_classification_fedavg \
    --gpu_mapping_file "experiments/distributed/transformer_exps/gpu_mapping.yaml" \
    --gpu_mapping_key mapping_ink-ron \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset 20news \
    --data_file "data/data_loaders/20news_data_loader.pkl" \
    --partition_file "data/partition/20news_partition.pkl" \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --epochs 1 \
    --output_dir "/tmp/20news_fedavg/" \
    --fp16 
    # 2> ${LOG_FILE} &
```

