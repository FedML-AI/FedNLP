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

## AGNews experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset agnews \
  --data_file data/data_loaders/agnews_data_loader.pkl \
  --partition_file data/partition/agnews_partition.pkl \
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
  --embedding_name '' \
  --embedding_file '' \
  --device '' \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

## Semeval_2010_task8 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset semeval_2010_task8 \
  --data_file data/data_loaders/semeval_2010_task8_data_loader.pkl \
  --partition_file data/partition/semeval_2010_task8_partition.pkl \
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
  --embedding_name '' \
  --embedding_file '' \
  --device '' \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

## Sentiment140 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset sentiment140 \
  --data_file data/data_loaders/sentiment_140_data_loader.pkl \
  --partition_file data/partition/sentiment_140_partition.pkl \
  --partition_method uniform \
  --hidden_size 300  \
  --num_layers 1 \
  --embedding_dropout 0 \
  --lstm_dropout 0.1 \
  --attention_dropout 0 \
  --batch_size 128 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 10 \
  --embedding_name '' \
  --embedding_file '' \
  --device '' \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

## SST-2 experiments
```
python -m experiments.centralized.bilstm_exps.main_text_classification \
  --model bilstm \
  --dataset sst_2 \
  --data_file data/data_loaders/sst_2_data_loader.pkl \
  --partition_file data/partition/sst_2_partition.pkl \
  --partition_method uniform \
  --hidden_size 100  \
  --num_layers 1 \
  --embedding_dropout 0.3 \
  --lstm_dropout 0.5 \
  --attention_dropout 0 \
  --batch_size 32 \
  --max_seq_len 32 \
  --embedding_length 300 \
  --lr 0.001 \
  --wd 0.0001 \
  --epochs 30 \
  --embedding_name '' \
  --embedding_file '' \
  --device '' \
  --do_remove_stop_words False \
  --do_remove_low_freq_words 0
```

## Experiment Results
| Dataset | Model | Embedding | Metrics | Metrics Value |
| ------- | ------ | ------- | ------- | ------- |
| 20news | BiLSTM | glove | Accuracy| 74%/[73.18](https://arxiv.org/pdf/1809.05679v3.pdf) |
| agnews | BiLSTM | glove | Accuracy| 91%/[91.7](https://arxiv.org/pdf/1808.09644v1.pdf) |
| semeval_2010_task8 | BiLSTM | glove | F1 | 78%/[82.7](https://www.aclweb.org/anthology/Y15-1009.pdf) |
| sentiment140 | BiLSTM | glove | Accuracy| 84% |
| sst_2 | BiLSTM | glove | Accuracy | 80%/[87.5](https://arxiv.org/pdf/1503.00075.pdf) |

## Experiment Parameters
| dataset | hidden_size | num_layers | dropout | embedding_dropout | batch_size | max_seq_len | embedding_length | optimizer | lr | wd | epochs |
| ------- | ------ | ------- | ------- | ------- | ------- |------- | ------- | ------- | ------- | ------- | ------- |
| 20news | 300 |  1 |  0.5 | 0 | 32 |  512 |  300 |  adam |  0.001 |  0.0001 |  30 |
| agnews | 300 |  1 |  0.1 | 0 | 128 |  128 |  300 |  adam |  0.001 |  0.0001 |  10 |
| semeval_2010_task8 | 300 | 1 |  0.5 | 0.3 |  10 |  100 |  300 |  adam |  0.001 |  0.0001 |  50 |
| sentiment140 | 300 |  1 | 0.1 |  0 |  128 |  32 |  300 |  adam |  0.001 |  0.0001 |  10 |
| sst_2 | 100 |  1 |  0.5 | 0.3 |  32 |  32 |  300 |  adam |  0.001 |  0.0005 |  30 |



