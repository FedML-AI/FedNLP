#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7
LR=${8}
DATASET=${9}
DATA_FILE=${10}
PARTITION_FILE=${11}
EMBEDDING_FILE=${12}
LSTM_DROPOUT=${13}
EMBEDDING_DROPOUT=${14}
WD=${15}
MAX_SEQ_LEN=${16}
REMOVE_WORD=${17}
REMOVE_LOW_FREQ_WORD=${18}
CI=${19}
FED_ALG=${20}
SERVER_LR=${21}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

if [ "$FED_ALG" = "fedavg" ]
then
  mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 experiments/distributed/bilstm_exps/main_fedavg.py \
    --gpu_num_per_server $GPU_NUM_PER_SERVER \
    --gpu_server_num $SERVER_NUM \
    --dataset $DATASET \
    --data_file $DATA_FILE \
    --partition_file $PARTITION_FILE \
    --embedding_file $EMBEDDING_FILE \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WD \
    --lstm_dropout $LSTM_DROPOUT \
    --embedding_dropout $EMBEDDING_DROPOUT \
    --max_seq_len $MAX_SEQ_LEN \
    --do_remove_stop_words $REMOVE_WORD \
    --do_remove_low_freq_words $REMOVE_LOW_FREQ_WORD \
    --ci $CI
elif [ "$FED_ALG" = "fedopt" ]
then
  mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 experiments/distributed/bilstm_exps/main_fedopt.py \
    --gpu_num_per_server $GPU_NUM_PER_SERVER \
    --gpu_server_num $SERVER_NUM \
    --dataset $DATASET \
    --data_file $DATA_FILE \
    --partition_file $PARTITION_FILE \
    --embedding_file $EMBEDDING_FILE \
    --client_num_in_total $CLIENT_NUM \
    --client_num_per_round $WORKER_NUM \
    --comm_round $ROUND \
    --epochs $EPOCH \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --server_lr $SERVER_LR \
    --wd $WD \
    --lstm_dropout $LSTM_DROPOUT \
    --embedding_dropout $EMBEDDING_DROPOUT \
    --max_seq_len $MAX_SEQ_LEN \
    --do_remove_stop_words $REMOVE_WORD \
    --do_remove_low_freq_words $REMOVE_LOW_FREQ_WORD \
    --ci $CI
else
  echo "no such federated algorithm!"
  exit 1
fi


