FL_ALG=$1
PARTITION_METHOD=$2
C_LR=$3
S_LR=$4
MU=$5
ROUND=$6

LOG_FILE="fedavg_transformer_se.log"
WORKER_NUM=5
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=mrqa
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_se \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_ink-titan-mrqa \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --ci $CI \
  --dataset "${DATA_NAME}" \
  --data_file "${DATA_DIR}/data_files/${DATA_NAME}_data.h5" \
  --partition_file "${DATA_DIR}/partition_files/${DATA_NAME}_partition.h5" \
  --partition_method $PARTITION_METHOD \
  --fl_algorithm $FL_ALG \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 50 \
  --eval_batch_size 128 \
  --max_seq_length 128 \
  --lr $C_LR \
  --server_lr $S_LR \
  --fedprox_mu $MU \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --fp16 --reprocess_input_data
  2> ${LOG_FILE} &


# sh run_span_extraction.sh FedAvg "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 50

# sh run_span_extraction.sh FedProx "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 50

# sh run_span_extraction.sh FedOPT "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 50