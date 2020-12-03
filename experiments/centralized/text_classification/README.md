## Experiment Tracking
```shell script
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```
## Run Experiments

## 20News experiments
```
sh run_text_classification.sh bilstm 20news ../../../data/data_loaders/20news_data_loader.pkl ../../../data/partition/20news_partition.pkl uniform 256 1 0.1 32 256 300 adam 0.001 0.0001 10
```

## AGNews experiments
```
sh run_text_classification.sh bilstm agnews ../../../data/data_loaders/agnews_data_loader.pkl ../../../data/partition/agnews_partition.pkl uniform 256 1 0.1 128 50 300 adam 0.001 0.0001 10
```

## Semeval_2010_task8 experiments
```
sh run_text_classification.sh bilstm semeval_2010_task8 ../../../data/data_loaders/semeval_2010_task8_data_loader.pkl ../../../data/partition/semeval_2010_task8_partition.pkl uniform 256 1 0.1 32 32 300 adam 0.001 0.0001 10
```

## Sentiment140 experiments
```
sh run_text_classification.sh bilstm sentiment140 ../../../data/data_loaders/sentiment140_data_loader.pkl ../../../data/partition/sentiment140_partition.pkl uniform 256 1 0.1 32 128 300 adam 0.001 0.0001 10
```

## SST-2 experiments
```
sh run_text_classification.sh bilstm sst_2 ../../../data/data_loaders/sst_2_data_loader.pkl ../../../data/partition/sst_2_partition.pkl uniform 256 1 0.1 128 12 300 adam 0.001 0.0001 10
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
| sst_2 | BiLSTM | random | 65.45% |
| sst_2 | BiLSTM+Attention | random | 65.29% |

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

semeval_2010_task8: macro-averaged F1-Score for (9+1)-way classification, taking directionality into account([BiLSTM:82.7, BiLSTM+Attention:84.0](https://arxiv.org/pdf/1901.08163v1.pdf))

sentiment140: Accuracy, F1(treat neutral as positive or negative)

sst_2: Accuracy, 5-class/binary classification ([BiLSTM: 87.5](https://arxiv.org/pdf/1910.03474v1.pdf))

