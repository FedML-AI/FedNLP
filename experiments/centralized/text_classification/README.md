## Run Experiments

## MNIST experiments
```
sh run_fednlp_classification_centralized_pytorch.sh bilstm_attention 20news "../../../data/data_loaders/20news_data_loader.pkl" "../../../../data/fednlp/partition/20news_partition_pkl" uniform 512 1 0.1 64 -1 300 adam 0.001 0.0001 5

```