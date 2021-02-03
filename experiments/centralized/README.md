# Centralized Training


## LSTM-based 

**TODO**

## Transformer-based 

```bash 
declare -a datasets=("20news" "agnews" "sentiment140" "sst_2")
for DATA_NAME in "${datasets[@]}"
do
  export CUDA_VISIBLE_DEVICES=0 \
  python -m experiments.centralized.transformer_exps.text_classification \
      --dataset ${DATA_NAME} \
      --data_file data/data_files/${DATA_NAME}_data.h5 \
      --partition_file data/partition_files/${DATA_NAME}_partition.h5 \
      --partition_method uniform \
      --model_type distilbert \
      --model_name distilbert-base-uncased \
      --do_lower_case True \
      --train_batch_size 32 \
      --eval_batch_size 32 \
      --max_seq_length 256 \
      --learning_rate 1e-5 \
      --num_train_epochs 5 \
      --output_dir /tmp/${DATA_NAME}_fed/ \
      --n_gpu 1 --fp16
done
```

## Question Answering (SQuAD)

```bash
export CUDA_VISIBLE_DEVICES=0 \
python -m experiments.centralized.transformer_exps.question_answering \
    --dataset squad_1.1 \
    --data_file data/data_files/squad_1.1_data.h5 \
    --partition_file data/partition_files/squad_1.1_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case True \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --max_seq_length 256 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --output_dir /tmp/squad_1.1/ \
    --fp16
```

## Named Entity Recognition

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.centralized.transformer_exps.named_entity_recognition \
    --dataset wikigold \
    --data_file data/data_files/wikigold_data.h5 \
    --partition_file data/partition_files/wikigold_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased \
    --do_lower_case False \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 128 \
    --learning_rate 4e-5 \
    --num_train_epochs 2 \
    --output_dir /tmp/wikigold/ \
    --fp16
```
