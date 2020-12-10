## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Experimental Tracking
```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments


### shakespeare experiments
```
sh experiments/distributed/bilstm_exps/run_fedavg_distributed_pytorch.sh 4 4 1 4 bilstm uniform 100 10 64 0.001 sentiment140 "data/data_loaders/sentiment_140_data_loader.pkl" "data/partition/sentiment_140_partition.pkl" 300 1 0.1 0 0 32 "data/pretrained/glove.840B.300d.txt" glove 300 False 5 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 4 4 1 4 bilstm uniform 100 10 64 0.001 sentiment140 "data/data_loaders/sentiment_140_data_loader.pkl" "data/partition/sentiment_140_partition.pkl" 300 1 0.1 0 0 32 "data/pretrained/glove.840B.300d.txt" glove 300 False 5 0 > ./fedavg-bilstm-sentiment140.txt 2>&1 &
```
