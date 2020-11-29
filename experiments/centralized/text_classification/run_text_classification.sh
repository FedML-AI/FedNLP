#!/usr/bin/env bash

MODEL=$1
DATASET=$2
DATA_FILE=$3
PARTITION_FILE=$4
PARTITION_METHOD=$5
HIDDEN_SIZE=$6
NUM_LAYERS=$7
DROPOUT=$8
BATCH_SIZE=$9
MAX_SEQ_LEN=${10}
EMBEDDING_LENGTH=${11}
OPTIMIZER=${12}
LR=${13}
WD=${14}
EPOCHS=${15}
EMBEDDING_NAME=${16}
EMBEDDING_FILE=${17}

python3 ./main_text_classification.py \
  --model $MODEL \
  --dataset $DATASET \
  --data_file $DATA_FILE \
  --partition_file $PARTITION_FILE \
  --partition_method $PARTITION_METHOD \
  --hidden_size $HIDDEN_SIZE  \
  --num_layers $NUM_LAYERS \
  --dropout $DROPOUT \
  --batch_size $BATCH_SIZE \
  --max_seq_len $MAX_SEQ_LEN \
  --embedding_length $EMBEDDING_LENGTH \
  --lr $LR \
  --wd $WD \
  --epochs $EPOCHS \
  --embedding_name $EMBEDDING_NAME \
  --embedding_file $EMBEDDING_FILE