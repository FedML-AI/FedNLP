FL_ALG=$1
PARTITION_METHOD=$2
C_LR=$3
S_LR=$4
ROUND=$5
WORKER_NUM=$6
LAYERS=$7

LOG_FILE="fedavg_transformer_tc.log"
# WORKER_NUM=10
CI=0

DATA_DIR=~/fednlp_data/
DATA_NAME=20news
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile mpi_host_file \
python -m fedavg_main_tc \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key mapping_a100 \
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
  --train_batch_size 32 \
  --eval_batch_size 8 \
  --max_seq_length 256 \
  --lr $C_LR \
  --server_lr $S_LR \
  --epochs 1 \
  --output_dir "/tmp/fedavg_${DATA_NAME}_output/" \
  --freeze_layers $LAYERS


# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=5.0" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=10.0" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "niid_label_clients=100_alpha=1.0" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "uniform" 5e-5 0.1 50
# sh run_text_classification.sh FedAvg "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 50

# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=5.0" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=10.0" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "niid_label_clients=100_alpha=1.0" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "uniform" 5e-5 0.1 50
# sh run_text_classification.sh FedProx "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 50

# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=5.0" 5e-5 0.1 50
# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=10.0" 5e-5 0.1 50
# sh run_text_classification.sh FedOPT "niid_label_clients=100_alpha=1.0" 5e-5 0.1 50
# sh run_text_classification.sh FedOPT "uniform" 5e-5 0.1 300
# sh run_text_classification.sh FedOPT "niid_quantity_clients=100_beta=5.0" 5e-5 0.1 50