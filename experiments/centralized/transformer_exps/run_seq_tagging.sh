DATA_NAME=onto
CUDA_VISIBLE_DEVICES=6 python -m experiments.centralized.transformer_exps.main_st \
    --dataset ${DATA_NAME} \
    --data_file ~/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ~/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --evaluate_during_training_steps 2000 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1