## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Experimental Tracking
```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments


### 20news experiments
```
sh experiments/distributed/bilstm_exps/run_fedavg_distributed_pytorch.sh 10 10 1 4 bilstm uniform 100 1 32 0.01 20news "data/data_loaders/20news_data_loader.pkl" "data/partition/20news_partition.pkl" 300 1 0.1 0 0 32 "data/pretrained/glove.840B.300d.txt" glove 300 True 5 0

##run on background
nohup sh experiments/distributed/bilstm_exps/run_fedavg_distributed_pytorch.sh 4 4 1 4 bilstm uniform 100 30 32 0.001 20news "data/data_loaders/20news_data_loader.pkl" "data/partition/20news_partition.pkl" 300 1 0.1 0 0 32 "data/pretrained/glove.840B.300d.txt" glove 300 True 5 0 > ./fedavg-bilstm-20news.txt 2>&1 &
```
