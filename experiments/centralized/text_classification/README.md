## Run Experiments

## MNIST experiments
```
sh run_text_classification_centralized_pytorch.sh bilstm_attention 20news ../../../data/data_loaders/20news_data_loader.pkl ../../../data/partition/20news_partition.pkl uniform 256 1 0.1 64 -1 100 adam 0.001 0.0001 5

```