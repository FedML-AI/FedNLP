LOG_FILE="fedavg_transformer_tc.log"
CLIENT_NUM=10
WORKER_NUM=10
ROUND=500
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m text_classification_fedavg \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_default \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset sentiment140 \
  --data_file "../../../data/text_classification/Sentiment140/sentiment_140_data_loader.pkl" \
  --partition_file "../../../data/text_classification/Sentiment140/sentiment_140_partition.pkl" \
  --partition_method uniform \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --max_seq_length 128 \
  --learning_rate 1e-5 \
  --epochs 1 \
  --output_dir "./output" \
  --fp16
  # 2> ${LOG_FILE} &