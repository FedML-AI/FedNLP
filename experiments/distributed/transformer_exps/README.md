```
pip install setproctitle mpi4py
pip install paho-mqtt 
```


```bash
CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
ROUND=500
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
HOST_FILE=experiments/distributed/transformer_exps/mpi_host_file
hostname > HOST_FILE

# CUDA_VISIBLE_DEVICES=4,5,6,7 
mpirun -np $PROCESS_NUM -hostfile $HOST_FILE \
python -m experiments.distributed.transformer_exps.text_classification_fedavg \
    --gpu_num_per_server $GPU_NUM_PER_SERVER \
    --gpu_server_num $SERVER_NUM \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --ci $CI \
    --dataset_name 20news \
    --data_file data/data_loaders/20news_data_loader.pkl \
    --partition_file data/partition/20news_partition.pkl \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_seq_length 128 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/20news_fedavg/ \
    --fp16
```