## Experimental Tracking
```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments

## MNIST experiments
```
sh run_text_classification.sh bilstm_attention 20news ../../../data/data_loaders/20news_data_loader.pkl ../../../data/partition/20news_partition.pkl uniform 512 1 0.1 32 1000 300 adam 0.001 0.0001 10 word2vec ../../../data/pretrained/GoogleNews-vectors-negative300.bin
```

## Experiment Results
| Dataset | Model | Accuracy |
| ------- | ------ | ------- |
| 20news | BiLSTM+Attention | 62.76% |