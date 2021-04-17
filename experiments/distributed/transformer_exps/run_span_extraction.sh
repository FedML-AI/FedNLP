LOG_FILE="fedavg_transformer_se.log"
WORKER_NUM=10
ROUND=10  # 50 to test the simulated sampling
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=squad_1.1
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_se \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_config2_11 \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method uniform \
  --fl_algorithm "FedAvg" \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --max_seq_length 128 \
  --lr 1e-5 \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --fp16
  2> ${LOG_FILE} &
