## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

### shakespeare experiments
```
sh run_fedavg_distributed_pytorch.sh 2 2 1 4 rnn hetero 100 10 10 0.8 shakespeare "./../../../data/shakespeare" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 rnn hetero 100 10 10 0.8 shakespeare "./../../../data/shakespeare" 0 > ./fedavg-rnn-shakespeare.txt 2>&1 &
```
