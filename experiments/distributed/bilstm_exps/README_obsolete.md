## Run Experiments


### 20news experiments
#### FedOpt
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 500 1 32 0.005 20news "data/data_loaders_lower_bound/20news_data_loader.pkl" "data/partition_lower_bound/20news_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.5 0 0.0001 512 True 5 0 fedopt 0.1

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 500 1 32 0.005 20news "data/data_loaders_lower_bound/20news_data_loader.pkl" "data/partition_lower_bound/20news_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.5 0 0.0001 512 True 5 0 fedopt 0.1 > ./fedopt-20news.txt 2>&1 &
```

#### FedAvg
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 100 1 32 0.005 20news "data/data_loaders_lower_bound/20news_data_loader.pkl" "data/partition_lower_bound/20news_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.5 0 0.0001 512 True 5 0 fedavg

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 100 1 32 0.005 20news "data/data_loaders_lower_bound/20news_data_loader.pkl" "data/partition_lower_bound/20news_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.5 0 0.0001 512 True 5 0 fedavg > ./fedavg-20news.txt 2>&1 &
```

### AGNews experiments
#### FedOpt
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 200 1 128 0.005 agnews "data/data_loaders_lower_bound/agnews_data_loader.pkl" "data/partition_lower_bound/agnews_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.1 0 0.0001 128 False 0 0 fedopt 0.1

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 200 1 128 0.005 agnews "data/data_loaders_lower_bound/agnews_data_loader.pkl" "data/partition_lower_bound/agnews_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.1 0 0.0001 128 False 0 0 fedopt 0.1 > ./fedopt-agnews.txt 2>&1 &
```

#### FedAvg
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 60 1 128 0.005 agnews "data/data_loaders_lower_bound/agnews_data_loader.pkl" "data/partition_lower_bound/agnews_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.1 0 0.0001 128 False 0 0 fedavg

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 60 1 128 0.005 agnews "data/data_loaders_lower_bound/agnews_data_loader.pkl" "data/partition_lower_bound/agnews_partition.pkl" "data/pretrained/glove.6B.300d.txt" 0.1 0 0.0001 128 False 0 0 fedavg > ./fedavg-agnews.txt 2>&1 &
```


### Semeval_2010_task8 experiments
#### FedOpt
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 400 1 10 0.005 semeval_2010_task8 "data/data_loaders_lower_bound/semeval_2010_task8_data_loader.pkl" "data/partition_lower_bound/semeval_2010_task8_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0001 100 False 0 0 fedopt 0.1

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 400 1 10 0.005 semeval_2010_task8 "data/data_loaders_lower_bound/semeval_2010_task8_data_loader.pkl" "data/partition_lower_bound/semeval_2010_task8_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0001 100 False 0 0 fedopt 0.1 > ./fedopt-semeval-2010-task8.txt 2>&1 &
```

#### FedAvg
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 60 1 10 0.005 semeval_2010_task8 "data/data_loaders_lower_bound/semeval_2010_task8_data_loader.pkl" "data/partition_lower_bound/semeval_2010_task8_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0001 100 False 0 0 fedavg

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 60 1 10 0.005 semeval_2010_task8 "data/data_loaders_lower_bound/semeval_2010_task8_data_loader.pkl" "data/partition_lower_bound/semeval_2010_task8_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0001 100 False 0 0 fedavg > ./fedavg-semeval-2010-task8.txt 2>&1 &
```

### Sentiment140 experiments
#### FedOpt
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 100 1 256 0.001 sentiment140 "data/data_loaders_lower_bound/sentiment_140_data_loader.pkl" "data/partition_lower_bound/sentiment_140_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 5 0 fedopt 0.1

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 100 1 256 0.001 sentiment140 "data/data_loaders_lower_bound/sentiment_140_data_loader.pkl" "data/partition_lower_bound/sentiment_140_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 5 0 fedopt 0.1 > ./fedopt-sentiment140.txt 2>&1 &
```

#### FedAvg
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 50 1 256 0.001 sentiment140 "data/data_loaders_lower_bound/sentiment_140_data_loader.pkl" "data/partition_lower_bound/sentiment_140_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 5 0 fedavg

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 50 1 256 0.001 sentiment140 "data/data_loaders_lower_bound/sentiment_140_data_loader.pkl" "data/partition_lower_bound/sentiment_140_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 5 0 fedavg > ./fedavg-sentiment140.txt 2>&1 &
```

### SST-2 experiments
#### FedOpt
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 400 1 32 0.005 sst_2 "data/data_loaders_lower_bound/sst_2_data_loader.pkl" "data/partition_lower_bound/sst_2_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 0 0 fedopt 0.1

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 400 1 32 0.005 sst_2 "data/data_loaders_lower_bound/sst_2_data_loader.pkl" "data/partition_lower_bound/sst_2_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 0 0 fedopt 0.1 > ./fedopt-sst-2.txt 2>&1 &
```

#### FedAvg
```
sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 100 1 32 0.005 sst_2 "data/data_loaders_lower_bound/sst_2_data_loader.pkl" "data/partition_lower_bound/sst_2_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 0 0 fedavg

##run on background
nohup sh experiments/distributed/bilstm_exps/run_bilstm_exps.sh 10 10 1 4 100 1 32 0.005 sst_2 "data/data_loaders_lower_bound/sst_2_data_loader.pkl" "data/partition_lower_bound/sst_2_partition.pkl" "data/pretrained/glove.840B.300d.txt" 0.5 0.3 0.0005 32 False 0 0 fedavg > ./fedavg-sst-2.txt 2>&1 &
```