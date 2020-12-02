# FedNLP: A Benchmarking Framework for Federated Learning in Natural Language Processing
 
<!-- This is FedNLP, an application ecosystem for federated natural language processing based on FedML framework (https://github.com/FedML-AI/FedML). -->

FedNLP is a research-oriented benchmarking framework for advancing *federated learning* (FL) in *natural language processing* (NLP).
It uses FedML repository as the git submodule. In other words, FedNLP only focuses on adavanced models and dataset, while FedML supports various
federated optimizers (e.g., FedAvg) and platforms (Distributed Computing, IoT/Mobile, Standalone).

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fednlp python=3.7
conda activate fednlp
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n fednlp
pip install -r requirements.txt 
cd FedML; git submodule init; git submodule update; cd ../;
```



## Data Preparation

### Option 1: use our bash scripts to download the dataset that you need manually. 
For example, in the `data/text_classification/20Newsgroups` folder, you can  run `sh download_and_unzip.sh` to get the raw data and then process it following [data_preprocessing/README.md](data_preprocessing/README.md).

### Option 2: download our processed files from Google Drive.
Dwnload files for each dataset in this [link](https://drive.google.com/folderview?id=1OhZ5NDaVz0VZX5jy8V_I_sfR25R2k_OE).  
We provide two files for eac dataset: data files are saved in  **data_loaders**, and partition files are in directory **partition**. You need to put the downloaded `data_loaders` and `partition` in the `data` folder here. Simply put, we will have `data/data_loaders/*_data_loader.pkl` and `data/partition/*_partition.pkl` in the end.


## Experiments for Centralized Learning (Sanity Check)

### LSTM-based models 

For example, you can run the centralized version of the models for text classification tasks with BLSTM models ([model/bilstm.py](model/bilstm.py)).
```bash
sh experiments/centralized/text_classification/run_text_classification.sh \
    bilstm_attention 20news \
    data/data_loaders/20news_data_loader.pkl \
    data/partition/20news_partition.pkl \
    uniform 256 1 0.1 32 256 300 adam 0.001 0.0001 200 \
    word2vec data/pretrained/GoogleNews-vectors-negative300.bin
```

For more experiments, please read [experiments/centralized/README.md](experiments/centralized/README.md).

### Transformer-based models 

```bash
# Test the environment for the fed_transformers
python -m model.fed_transformers.test
```
**TODO**

## Experiments for Federated Learning

1. Shakespeare + BiLSTM + FedAvg:
```bash
sh experiments/distributed/text_classification/run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 100 1 10 0.8 shakespeare "./data/text_classification/shakespeare/" 0

##run on background
nohup sh experiments/distributed/text_classification/run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 100 1 10 0.8 shakespeare "./data/text_classification/shakespeare/" 0  2>&1 &
```

### Update FedML Submodule 
```
cd FedML
git checkout master && git pull
cd ..
git add FedML
git commit -m "updating submodule FedML to latest"
git push
``` 

## Code Structure of FedNLP
<!-- Note: The code of FedNLP only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda, without the need to use Git submodule. -->

- `FedML`: a soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.


- `data`: provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms.
FedNLP supports more advanced datasets and models.

- `data_preprocessing`: data loaders

- `model`: advanced NLP models.

- `trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments/distributed`: 
1. `experiments` is the entry point for training. It contains experiments in different platforms. We start from `distributed`.
1. Every experiment integrates FOUR building blocks `FedML` (federated optimizers), `data_preprocessing`, `model`, `trainer`.
3. To develop new experiments, please refer the code at `experiments/distributed/text-classification`.

- `experiments/centralized`: 
1. please provide centralized training script in this directory. 
2. This is used to get the reference model accuracy for FL. 
3. You may need to accelerate your training through distributed training on multi-GPUs and multi-machines. Please refer the code at `experiments/centralized/DDP_demo`.




## Citation
Please cite our FedNLP and FedML paper if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedNLP [1] and FedML [2]".

 
