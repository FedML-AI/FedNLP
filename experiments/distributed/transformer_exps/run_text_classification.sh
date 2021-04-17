LOG_FILE="fedavg_transformer_tc.log"
WORKER_NUM=10
ROUND=30  # 50 to test the simulated sampling
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=20news
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_tc \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_config2_11 \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method "niid_label_clients=100_alpha=5.0" \
  --fl_algorithm "FedProx" \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --max_seq_length 128 \
  --lr 5e-5 \
  --server_lr 0.1 \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/"


  # niid_label_clients=100.0_alpha=5.0