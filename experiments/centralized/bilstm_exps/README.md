## Experiment Tracking
```shell script
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments
Note that we always assume users will run the following scripts at root directory
### 20News experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset 20news \
  --data_file data/data_files/20news_data.h5 \
  --partition_file data/partition_files/20news_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 32 \
  --max_seq_len 512 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 30 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.6B.300d.txt \
  --device cuda:0 \
  --do_remove_stop_words True \
  --do_remove_low_freq_words 5
```

### AGNews experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset agnews \
  --data_file data/data_files/agnews_data.h5 \
  --partition_file data/partition_files/agnews_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0 \
  --lstm_dropout 0.1 \
  --attention_dropout 0 \
  --batch_size 128 \
  --max_seq_len 128 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 10 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.6B.300d.txt \
  --device cuda:0 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

### Semeval_2010_task8 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset semeval_2010_task8 \
  --data_file data/data_files/semeval_2010_task8_data.h5 \
  --partition_file data/partition_files/semeval_2010_task8_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 10 \
  --max_seq_len 100 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 30 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.840B.300d.txt \
  --device cuda:1 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

### Sentiment140 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset sentiment140 \
  --data_file data/data_files/sentiment_140_data.h5 \
  --partition_file data/partition_files/sentiment_140_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 128 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0005 \
  --epochs 10 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.840B.300d.txt \
  --device cuda:2 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 5
```

### SST-2 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset sst_2 \
  --data_file data/data_files/sst_2_data.h5 \
  --partition_file data/partition_files/sst_2_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 32 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0005 \
  --epochs 50 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.840B.300d.txt \
  --device cuda:1 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

## Run Lower Bound Experiments
### 20News experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification_lower_bound \
  --model bilstm \
  --dataset 20news \
  --data_file data/data_files_lower_bound/20news_data.h5 \
  --partition_file data/partition_files_lower_bound/20news_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 32 \
  --max_seq_len 512 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 50 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.6B.300d.txt \
  --device cuda:0 \
  --do_remove_stop_words True \
  --do_remove_low_freq_words 5
```

### AGNews experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification_lower_bound \
  --model bilstm \
  --dataset agnews \
  --data_file data/data_files_lower_bound/agnews_data.h5 \
  --partition_file data/partition_files_lower_bound/agnews_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0 \
  --lstm_dropout 0.1 \
  --attention_dropout 0 \
  --batch_size 128 \
  --max_seq_len 128 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 20 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.6B.300d.txt \
  --device cuda:0 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

### Semeval_2010_task8 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification_lower_bound \
  --model bilstm \
  --dataset semeval_2010_task8 \
  --data_file data/data_files_lower_bound/semeval_2010_task8_data.h5 \
  --partition_file data/partition_files_lower_bound/semeval_2010_task8_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 10 \
  --max_seq_len 100 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 50 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.840B.300d.txt \
  --device cuda:1 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

### Sentiment140 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification_lower_bound \
  --model bilstm \
  --dataset sentiment140 \
  --data_file data/data_files_lower_bound/sentiment_140_data.h5 \
  --partition_file data/partition_files_lower_bound/sentiment_140_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 256 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0005 \
  --epochs 15 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.840B.300d.txt \
  --device cuda:2 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 5
```

### SST-2 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification_lower_bound \
  --model bilstm \
  --dataset sst_2 \
  --data_file data/data_files_lower_bound/sst_2_data.h5 \
  --partition_file data/partition_files_lower_bound/sst_2_partition.h5 \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 32 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0005 \
  --epochs 50 \
  --embedding_name glove \
  --embedding_file data/pretrained/glove.840B.300d.txt \
  --device cuda:1 \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

## Experiment Results
| Dataset | Model | Embedding | Metrics | Metrics Value |
| ------- | ------ | ------- | ------- | ------- |
| 20news | BiLSTM | glove | Accuracy| 78%/[73.18%](https://arxiv.org/pdf/1809.05679v3.pdf) |
| agnews | BiLSTM | glove | Accuracy| 91.5%/[91.7%](https://arxiv.org/pdf/1808.09644v1.pdf) |
| semeval_2010_task8 | BiLSTM | glove | Accuracy,F1 | 74%,79%/[82.7%](https://www.aclweb.org/anthology/Y15-1009.pdf) |
| sentiment140 | BiLSTM | glove | Accuracy| 84% |
| sst_2 | BiLSTM | glove | Accuracy | 85.5%/[87.5%](https://arxiv.org/pdf/1503.00075.pdf) |

20news lower bound test eval accuracy statistics
all eval accuracy 45.493467524900275,41.927083338721324,44.6327683723579,38.828566389568785,42.03742938930706,41.909427966101696,40.024717524900275,42.337570626856916,50.092690677966104,43.94862288135593
Mean eval accuracy: 43.12
Maximum eval accuracy: 50.09
Minimum eval accuracy: 38.83
Median eval accuracy: 42.19
Pvariance of eval accuracy: 8.99
Pstdev of eval accuracy: 3.00

agnews lower bound test eval accuracy statistics
all eval accuracy 88.81076393127441,89.24913202921549,88.828125,89.25347226460775,89.17534726460775,88.61979166666667,88.83680559794108,89.2578125,89.02777786254883,88.81944452921549
Mean eval accuracy: 88.99
Maximum eval accuracy: 89.26
Minimum eval accuracy: 88.62
Median eval accuracy: 88.93
Pvariance of eval accuracy: 0.05
Pstdev of eval accuracy: 0.22

semeval_2010_task8 lower bound test eval accuracy statistics
all eval accuracy 44.1491596698761,47.77310924670275,45.877100846346686,46.42857143458198,45.81407563826617,47.43697479191948,46.42857143458198,49.32773110445808,45.420168070232165,45.65651261105257
Mean eval accuracy: 46.43
Maximum eval accuracy: 49.33
Minimum eval accuracy: 44.15
Median eval accuracy: 46.15
Pvariance of eval accuracy: 1.87
Pstdev of eval accuracy: 1.37

sentiment140 lower bound test eval accuracy statistics
all eval accuracy 84.25716400146484,81.6842041015625,81.63900756835938,82.30403900146484,82.05223083496094,83.86653900146484,81.91341400146484,83.4420166015625,82.51065063476562,82.48805236816406
Mean eval accuracy: 82.62
Maximum eval accuracy: 84.26
Minimum eval accuracy: 81.64
Median eval accuracy: 82.40
Pvariance of eval accuracy: 0.77
Pstdev of eval accuracy: 0.88

sst_2 lower bound test eval accuracy statistics
all eval accuracy 78.54091041966488,78.74319417853104,79.6752117223907,79.62605861195347,80.02117357755962,77.90381127073054,79.44457044099507,80.55240470484684,79.39541746440686,80.66205382765385
Mean eval accuracy: 79.46
Maximum eval accuracy: 80.66
Minimum eval accuracy: 77.90
Median eval accuracy: 79.54
Pvariance of eval accuracy: 0.68
Pstdev of eval accuracy: 0.83

## Experiment Parameters
| dataset | hidden_size | num_layers | dropout | embedding_dropout | batch_size | max_seq_len | embedding_length | optimizer | lr | wd | epochs | time(s) |
| ------- | ------ | ------- | ------- | ------- | ------- |------- | ------- | ------- | ------- | ------- | ------- | ------- 
| 20news | 300 |  1 |  0.5 | 0 | 32 |  512 |  300 |  adam |  0.001 |  0.0001 |  30 | 1952 |
| agnews | 300 |  1 |  0.1 | 0 | 128 |  128 |  300 |  adam |  0.001 |  0.0001 | 10 | 1033 |
| semeval_2010_task8 | 300 | 1 |  0.5 | 0.3 |  10 |  100 |  300 |  adam |  0.001 |  0.0001 | 30 | 560 |
| sentiment140 | 300 |  1 | 0.5 |  0.3 |  256 |  32 |  300 |  adam |  0.001 |  0.0005 | 10 | 6029 |
| sst_2 | 100 |  1 |  0.5 | 0.3 |  32 |  32 |  300 |  adam |  0.001 |  0.0005 | 50 | 291 |




