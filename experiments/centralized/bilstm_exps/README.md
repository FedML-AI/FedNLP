## Experiment Tracking
```shell script
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments
Note that we always assume users will run the following scripts at root directory
## 20News experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset 20news \
  --data_file data/data_loaders/20news_data_loader.pkl \
  --partition_file data/partition/20news_partition.pkl \
  --partition_method uniform \
  --hidden_size 256  \
  --num_layers 1 \
  --dropout 0.1 \
  --batch_size 32 \
  --max_seq_len 256 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 100 \
  --embedding_name '' \
  --embedding_file '' \
  --device ''
```

## AGNews experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset agnews \
  --data_file data/data_loaders/agnews_data_loader.pkl \
  --partition_file data/partition/agnews_partition.pkl \
  --partition_method uniform \
  --hidden_size 256  \
  --num_layers 1 \
  --dropout 0.1 \
  --batch_size 128 \
  --max_seq_len 50 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 100 \
  --embedding_name '' \
  --embedding_file '' \
  --device ''
```

## Semeval_2010_task8 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset semeval_2010_task8 \
  --data_file data/data_loaders/semeval_2010_task8_data_loader.pkl \
  --partition_file data/partition/semeval_2010_task8_partition.pkl \
  --partition_method uniform \
  --hidden_size 256  \
  --num_layers 1 \
  --dropout 0.1 \
  --batch_size 32 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 100 \
  --embedding_name '' \
  --embedding_file '' \
  --device ''
```

## Sentiment140 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset sentiment140 \
  --data_file data/data_loaders/sentiment_140_data_loader.pkl \
  --partition_file data/partition/sentiment_140_partition.pkl \
  --partition_method uniform \
  --hidden_size 256  \
  --num_layers 1 \
  --dropout 0.1 \
  --batch_size 32 \
  --max_seq_len 128 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 100 \
  --embedding_name '' \
  --embedding_file '' \
  --device ''
```

## SST-2 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset sst_2 \
  --data_file data/data_loaders/sst_2_data_loader.pkl \
  --partition_file data/partition/sst_2_task8_partition.pkl \
  --partition_method uniform \
  --hidden_size 256  \
  --num_layers 1 \
  --dropout 0.1 \
  --batch_size 128 \
  --max_seq_len 12 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 100 \
  --embedding_name '' \
  --embedding_file '' \
  --device ''
```

## Experiment Results
| Dataset | Model | Embedding | Accuracy |
| ------- | ------ | ------- | ------- |
| 20news | BiLSTM | word2vec+freeze | 57.91% |
| 20news | BiLSTM | word2vec+unfreeze | 56.78% |
| 20news | BiLSTM | glove+freeze | 66.10% |
| 20news | BiLSTM | glove+unfreeze | 64.42% |
| 20news | BiLSTM | random | 66.76% |
| 20news | BiLSTM+Attention | random | 63.97% |
| agnews | BiLSTM | random | 90.46% |
| agnews | BiLSTM+Attention | random | 89.96% |
| semeval_2010_task8 | BiLSTM | random | 69.68% |
| semeval_2010_task8 | BiLSTM+Attention | random | 68.79% |
| sentiment140 | BiLSTM | random | 60.64% |
| sentiment140 | BiLSTM+Attention | random | 60.04% |
| sst_2 | BiLSTM | glove | 82.07% |

## Experiment Parameters
| dataset | hidden_size | num_layers | dropout | batch_size | max_seq_len | embedding_length | optimizer | lr | wd | epochs |
| ------- | ------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 20news | 256 |  1 |  0.1 |  32 |  256 |  300 |  adam |  0.001 |  0.0001 |  200 |
| agnews | 256 |  1 |  0.1 |  128 |  50 |  300 |  adam |  0.001 |  0.0001 |  200 |
| semeval_2010_task8 | 256 |  1 |  0.1 |  32 |  32 |  300 |  adam |  0.001 |  0.0001 |  200 |
| sentiment140 | 256 |  1 |  0.1 |  128 |  32 |  300 |  adam |  0.001 |  0.0001 |  200 |
| sst_2 | 256 |  1 |  0.1 |  128 |  12 |  300 |  adam |  0.001 |  0.0001 |  200 |


## Experiment Metrics
20News: Accuracy, Precision, Recall, F1

agnews: Error Rate, Accuracy([BiLSTM:91.7](https://arxiv.org/pdf/1808.09644v1.pdf))

semeval_2010_task8: macro-averaged F1-Score for (9+1)-way classification, taking directionality into account([BiLSTM:82.7](https://www.aclweb.org/anthology/Y15-1009.pdf), 
[BiLSTM+Attention:84.0](https://www.aclweb.org/anthology/P16-2034.pdf))

sentiment140: Accuracy, F1(treat neutral as positive or negative)

sst_2: Accuracy, binary classification ([BiLSTM: 87.5](https://arxiv.org/pdf/1910.03474v1.pdf))

