#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
SERVER_NUM=$3
GPU_NUM_PER_SERVER=$4
MODEL=$5
DISTRIBUTION=$6
ROUND=$7
EPOCH=$8
BATCH_SIZE=$9
LR=${10}
DATASET=${11}
DATA_FILE=${12}
PARTITION_FILE=${13}
HIDDEN_SIZE=${14}
NUM_LAYERS=${15}
LSTM_DROPOUT=${16}
EMBEDDING_DROPOUT=${17}
ATTENTION_DROPOUT=${18}
MAX_SEQ_LEN=${19}
EMBEDDING_FILE=${20}
EMBEDDING_NAME=${21}
EMBEDDING_LENGTH=${22}
REMOVE_WORD=${23}
REMOVE_LOW_FREQ_WORD=${24}
CI=${25}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 experiments/distributed/text_classification/main_fedavg.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_file $DATA_FILE \
  --partition_file $PARTITION_FILE \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --hidden_size $HIDDEN_SIZE \
  --num_layers $NUM_LAYERS \
  --lstm_dropout $LSTM_DROPOUT \
  --embedding_dropout $EMBEDDING_DROPOUT \
  --attention_dropout $ATTENTION_DROPOUT \
  --max_seq_len $MAX_SEQ_LEN \
  --embedding_file $EMBEDDING_FILE \
  --embedding_name $EMBEDDING_NAME \
  --embedding_length $EMBEDDING_LENGTH \
  --do_remove_stop_words $REMOVE_WORD \
  --do_remove_low_freq_words $REMOVE_LOW_FREQ_WORD \
  --ci $CI