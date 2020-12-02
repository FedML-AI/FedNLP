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
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 100 1 10 0.8 shakespeare "./../../../data/text_classification/shakespeare/" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 100 1 10 0.8 shakespeare "./../../../data/text_classification/shakespeare/" 0 > ./fedavg-rnn-shakespeare.txt 2>&1 &
```
