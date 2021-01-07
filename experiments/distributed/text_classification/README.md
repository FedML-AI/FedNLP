## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Experimental Tracking
```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments


### shakespeare experiments
```
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 100 1 10 0.8 shakespeare "./../../../data/text_classification/shakespeare/" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 100 1 10 0.8 shakespeare "./../../../data/text_classification/shakespeare/" 0 > ./fedavg-rnn-shakespeare.txt 2>&1 &
```

```bash
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 experiments/distributed/text_classification/main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI
```

